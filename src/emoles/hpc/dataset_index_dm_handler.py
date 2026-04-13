import json
import os
import random
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

import lmdb
import numpy as np
from ase.db import connect
from pyscf.scf.hf import dip_moment


def _prepend_env_path(env_name, path_entries):
    existing = [item for item in os.environ.get(env_name, "").split(os.pathsep) if item]
    prefix = []
    for entry in path_entries:
        if not entry or not os.path.exists(entry):
            continue
        if entry not in prefix and entry not in existing:
            prefix.append(entry)
    os.environ[env_name] = os.pathsep.join(prefix + existing)


def _prepare_runtime_environment():
    _prepend_env_path(
        "PATH",
        [
            "/opt/mamba/bin",
            "/root/Multiwfn_3.8_dev_bin_Linux_noGUI",
            "/root/software/mokit/bin",
        ],
    )
    mokit_python_root = "/root/software/mokit"
    if os.path.isdir(mokit_python_root) and mokit_python_root not in sys.path:
        sys.path.insert(0, mokit_python_root)
    _prepend_env_path("PYTHONPATH", [mokit_python_root])


_prepare_runtime_environment()


def _install_emoles_pyscf_stub():
    if "emoles.pyscf" in sys.modules:
        return

    stub = types.ModuleType("emoles.pyscf")

    def get_dipole_info(mol, dm):
        return np.array(dip_moment(mol, dm, unit="DEBYE"), dtype=float)

    stub.get_dipole_info = get_dipole_info
    sys.modules["emoles.pyscf"] = stub


_install_emoles_pyscf_stub()

import emoles.inference.postprocess as postprocess_module
from emoles.inference.postprocess import dm_infer_light_entry_from_lmdb


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _install_task_stdout_tee(log_filename="task.stdout"):
    log_path = Path(log_filename).resolve()
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_handle)
    sys.stderr = _TeeStream(sys.__stderr__, log_handle)
    return log_handle


def _log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[dataset-index-handler {timestamp}] {message}", flush=True)


def _run_preflight_checks(config):
    require_multiwfn = bool(config.get("calc_esp_flag", True) or config.get("calc_electronic_flag", True))
    binary_checks = {
        "python": sys.executable,
        "Multiwfn": shutil.which("Multiwfn") if require_multiwfn else None,
    }
    module_checks = {}
    missing = []
    for name in ["ase", "lmdb", "pyscf", "emoles", "mokit.lib.py2fch_direct"]:
        try:
            module = __import__(name, fromlist=["*"])
            module_checks[name] = getattr(module, "__file__", "built-in")
        except Exception as exc:
            module_checks[name] = f"ERR: {exc!r}"
            missing.append(name)

    _log(f"preflight executable={sys.executable}")
    _log(f"preflight PATH={os.environ.get('PATH', '')}")
    _log(f"preflight PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    _log(f"preflight binaries={json.dumps(binary_checks, indent=2)}")
    _log(f"preflight modules={json.dumps(module_checks, indent=2)}")

    if require_multiwfn and binary_checks["Multiwfn"] is None:
        missing.append("Multiwfn")

    if missing:
        raise RuntimeError(f"Preflight missing_or_broken: {sorted(set(missing))}")


def _safe_file_size(path):
    try:
        return int(Path(path).stat().st_size)
    except FileNotFoundError:
        return None


def _format_eta(seconds):
    if seconds is None:
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{rem:04.1f}s"


def _render_progress(desc, current, total, started_at):
    elapsed = max(0.0, time.perf_counter() - started_at)
    total = max(0, int(total)) if total is not None else None
    current = max(0, int(current))
    if total:
        ratio = min(1.0, current / total)
        filled = int(ratio * 24)
        bar = "#" * filled + "-" * (24 - filled)
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (total - current) / rate if rate > 0 and current < total else 0.0
        return (
            f"[{desc}] |{bar}| {current}/{total} "
            f"({ratio * 100:5.1f}%) elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}"
        )
    return f"[{desc}] {current} elapsed={_format_eta(elapsed)}"


def _progress_iter(iterable, total=None, desc="progress", update_every=None):
    total = None if total is None else int(total)
    if update_every is None:
        if total and total > 0:
            update_every = max(1, min(100, total // 20))
        else:
            update_every = 50

    started_at = time.perf_counter()
    current = 0
    _log(f"{desc}: start (total={total if total is not None else 'unknown'})")
    for item in iterable:
        current += 1
        if current == 1 or current % update_every == 0 or (total is not None and current >= total):
            _log(_render_progress(desc, current, total, started_at))
        yield item
    _log(f"{desc}: done in {_format_eta(time.perf_counter() - started_at)}")


def _install_postprocess_progress_hook():
    def _wrapped_tqdm(iterable=None, *args, **kwargs):
        desc = kwargs.get("desc") or "dm_infer"
        total = kwargs.get("total")
        if iterable is None:
            return []
        return _progress_iter(iterable, total=total, desc=desc)

    postprocess_module.tqdm = _wrapped_tqdm


def _remove_sqlite_sidecars(db_path):
    for path in (db_path, f"{db_path}-shm", f"{db_path}-wal"):
        if os.path.exists(path):
            os.remove(path)


def _reset_lmdb_directory(lmdb_path):
    if os.path.isdir(lmdb_path):
        for root, dirs, files in os.walk(lmdb_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)


def _lmdb_key(index):
    return int(index).to_bytes(4, byteorder="big")


def _open_lmdb_env(lmdb_path, readonly=False):
    if readonly:
        return lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            subdir=True,
        )
    os.makedirs(lmdb_path, exist_ok=True)
    return lmdb.open(
        str(lmdb_path),
        map_size=1024**4,
        lock=True,
        subdir=True,
    )


def _resolve_random_shard_row_ids(
    total_rows,
    shard_id,
    n_shards,
    shuffle_seed,
    items_per_shard=None,
    max_source_rows=None,
):
    all_row_ids = list(range(1, int(total_rows) + 1))
    rng = random.Random(int(shuffle_seed))
    rng.shuffle(all_row_ids)

    if max_source_rows is not None:
        all_row_ids = all_row_ids[: min(len(all_row_ids), int(max_source_rows))]

    shard_id = int(shard_id)
    n_shards = int(n_shards)
    selection_mode = "random_full_cover"
    if items_per_shard is not None:
        items_per_shard = int(items_per_shard)
        shard_size = int(items_per_shard)
        start = min(shard_id * shard_size, len(all_row_ids))
        stop = min(start + shard_size, len(all_row_ids))
        selection_mode = "random_fixed_size"
    else:
        shard_size = (len(all_row_ids) + n_shards - 1) // n_shards
        start = min(shard_id * shard_size, len(all_row_ids))
        stop = min(start + shard_size, len(all_row_ids))
    shard_row_ids = sorted(all_row_ids[start:stop])
    return {
        "selected_row_ids": shard_row_ids,
        "selected_total": len(all_row_ids),
        "shard_size": int(shard_size),
        "start": int(start),
        "stop": int(stop),
        "selection_mode": selection_mode,
        "items_per_shard": items_per_shard,
    }


def _copy_selected_rows(source_db_path, shard_db_path, selected_row_ids):
    _remove_sqlite_sidecars(shard_db_path)

    selected = 0
    with connect(source_db_path) as src_db, connect(shard_db_path) as dump_db:
        total_rows = int(src_db.count())
        for row_id in _progress_iter(selected_row_ids, total=len(selected_row_ids), desc="copy-ase-db"):
            row = src_db.get(id=int(row_id))
            source_idx = int(row.id) - 1

            atoms = row.toatoms()
            data = dict(getattr(row, "data", None) or {})
            kvp = dict(getattr(row, "key_value_pairs", None) or {})
            dielectric_constant = getattr(row, "dielectric_constant", None)
            if dielectric_constant is not None and "dielectric_constant" not in data:
                data["dielectric_constant"] = dielectric_constant
            data["source_idx"] = int(source_idx)
            data["source_row_id"] = int(row.id)
            dump_db.write(atoms, key_value_pairs=kvp, data=data)
            selected += 1
    return {
        "selected": int(selected),
        "total_rows": int(total_rows),
        "row_id_min": int(min(selected_row_ids)) if selected_row_ids else None,
        "row_id_max": int(max(selected_row_ids)) if selected_row_ids else None,
    }


def _resolve_worker_lmdb_map(infer_root):
    worker_lmdb_map = {}
    for path in sorted(Path(infer_root).iterdir()):
        if not path.is_dir() or not path.name.endswith(".lmdb"):
            continue
        stem = path.stem
        worker_idx = int(stem.split("_")[-1])
        worker_lmdb_map[worker_idx] = path
    return worker_lmdb_map


def _extract_selected_lmdb(infer_root, target_lmdb_path, selected_row_ids):
    worker_lmdb_map = _resolve_worker_lmdb_map(infer_root)
    worker_count = len(worker_lmdb_map)
    _reset_lmdb_directory(target_lmdb_path)

    selected_by_worker = {}
    for row_id in selected_row_ids:
        worker_idx = (int(row_id) - 1) % worker_count
        selected_by_worker.setdefault(worker_idx, []).append(int(row_id))

    copied = 0
    missing = []
    fallback_lookups = 0
    output_env = _open_lmdb_env(target_lmdb_path, readonly=False)
    txn = output_env.begin(write=True)
    input_env_cache = {}
    try:
        for worker_idx, row_ids in sorted(selected_by_worker.items()):
            _log(f"copy-lmdb: worker_{worker_idx:02d} handles {len(row_ids)} selected rows")
            input_env = input_env_cache.get(worker_idx)
            if input_env is None:
                input_env = _open_lmdb_env(worker_lmdb_map[worker_idx], readonly=True)
                input_env_cache[worker_idx] = input_env

            with input_env.begin() as input_txn:
                for row_id in _progress_iter(
                    row_ids,
                    total=len(row_ids),
                    desc=f"copy-lmdb-worker-{worker_idx:02d}",
                ):
                    payload = input_txn.get(_lmdb_key(row_id))
                    if payload is None:
                        for fallback_idx, fallback_path in sorted(worker_lmdb_map.items()):
                            if fallback_idx == worker_idx:
                                continue
                            fallback_env = input_env_cache.get(fallback_idx)
                            if fallback_env is None:
                                fallback_env = _open_lmdb_env(fallback_path, readonly=True)
                                input_env_cache[fallback_idx] = fallback_env
                            with fallback_env.begin() as fallback_txn:
                                payload = fallback_txn.get(_lmdb_key(row_id))
                            if payload is not None:
                                fallback_lookups += 1
                                break
                    if payload is None:
                        missing.append(int(row_id))
                        continue
                    txn.put(_lmdb_key(row_id), payload)
                    copied += 1
                    if copied % 256 == 0:
                        txn.commit()
                        txn = output_env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        for input_env in input_env_cache.values():
            input_env.close()
        output_env.close()

    return {
        "copied": int(copied),
        "missing": missing,
        "worker_count": int(worker_count),
        "fallback_lookups": int(fallback_lookups),
    }


def main():
    task_stdout_handle = _install_task_stdout_tee()
    config_path = Path("task_config.json").resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _install_postprocess_progress_hook()

    bundle_root = Path(config["remote_bundle_root"]).resolve()
    source_db_path = str(bundle_root / "optimized_all.db")
    infer_root = Path(bundle_root / "infer")
    shard_db_path = str(Path("raw.db").resolve())
    shard_lmdb_path = Path("infer_selected.lmdb").resolve()
    results_dir = Path("results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_name": config["dataset_name"],
        "remote_bundle_root": str(bundle_root),
        "shard_id": int(config["shard_id"]),
        "n_shards": int(config["n_shards"]),
        "shuffle_seed": int(config.get("random_seed", config.get("shuffle_seed", 20260413))),
        "items_per_shard": config.get("items_per_shard"),
    }

    try:
        _run_preflight_checks(config)
        _log(
            "task-start "
            f"dataset={config['dataset_name']} shard={config['shard_id']}/{config['n_shards']} "
            f"items_per_shard={config.get('items_per_shard')} seed={payload['shuffle_seed']}"
        )
        _log(f"cwd={Path.cwd()}")
        _log(f"config_path={config_path}")
        _log(f"bundle_root={bundle_root}")
        with connect(source_db_path) as src_db:
            total_rows = int(src_db.count())
        _log(f"source-db rows={total_rows} path={source_db_path}")
        selection_meta = _resolve_random_shard_row_ids(
            total_rows=total_rows,
            shard_id=int(config["shard_id"]),
            n_shards=int(config["n_shards"]),
            shuffle_seed=int(config.get("random_seed", config.get("shuffle_seed", 20260413))),
            items_per_shard=config.get("items_per_shard"),
            max_source_rows=config.get("max_source_rows"),
        )
        _log(
            "selection "
            f"mode={selection_meta['selection_mode']} "
            f"selected={len(selection_meta['selected_row_ids'])} "
            f"start={selection_meta['start']} stop={selection_meta['stop']}"
        )
        if selection_meta["selected_row_ids"]:
            preview_first = selection_meta["selected_row_ids"][:5]
            preview_last = selection_meta["selected_row_ids"][-5:]
            _log(f"selection-preview first5={preview_first} last5={preview_last}")

        t0 = time.perf_counter()
        shard_meta = _copy_selected_rows(
            source_db_path=source_db_path,
            shard_db_path=shard_db_path,
            selected_row_ids=selection_meta["selected_row_ids"],
        )
        db_slice_seconds = time.perf_counter() - t0
        _log(
            "copy-ase-db done "
            f"rows={shard_meta['selected']} elapsed={_format_eta(db_slice_seconds)} "
            f"raw_db={shard_db_path}"
        )
        _log(f"raw_db_size={_safe_file_size(shard_db_path)} bytes")

        t0 = time.perf_counter()
        lmdb_meta = _extract_selected_lmdb(
            infer_root=infer_root,
            target_lmdb_path=shard_lmdb_path,
            selected_row_ids=selection_meta["selected_row_ids"],
        )
        lmdb_slice_seconds = time.perf_counter() - t0
        _log(
            "copy-lmdb done "
            f"copied={lmdb_meta['copied']} missing={len(lmdb_meta['missing'])} "
            f"fallback={lmdb_meta['fallback_lookups']} elapsed={_format_eta(lmdb_slice_seconds)} "
            f"lmdb={shard_lmdb_path}"
        )
        _log(f"infer_selected_data_mdb_size={_safe_file_size(shard_lmdb_path / 'data.mdb')} bytes")

        t0 = time.perf_counter()
        _log("infer-start running dm_infer_light_entry_from_lmdb")
        summary = dm_infer_light_entry_from_lmdb(
            abs_ase_path=shard_db_path,
            infer_lmdb_path=str(shard_lmdb_path),
            results_folder_path=str(results_dir),
            matrix_field=config.get("matrix_field", "hamiltonian"),
            convention=config.get("convention", "def2svp"),
            mol_charge=config.get("mol_charge", 0),
            transform_dm_flag=config.get("transform_dm_flag", True),
            calc_esp_flag=config.get("calc_esp_flag", True),
            calc_electronic_flag=config.get("calc_electronic_flag", True),
            max_items=config.get("max_items"),
            unified_pcm_flag=config.get("unified_pcm_flag", True),
            summary_json_name=config.get("summary_json_name", "merged_inference_summary.json"),
            updated_ase_db_path=config.get("updated_ase_db_path", "auto"),
        )
        dm_infer_seconds = time.perf_counter() - t0
        _log(
            "infer-done "
            f"successful_items={len(summary)} elapsed={_format_eta(dm_infer_seconds)} "
            f"results_dir={results_dir}"
        )
        payload.update(
            {
                "success": True,
                "selected_total_rows": int(selection_meta["selected_total"]),
                "selected_rows": int(shard_meta["selected"]),
                "source_total_rows": int(shard_meta["total_rows"]),
                "row_id_min": shard_meta["row_id_min"],
                "row_id_max": shard_meta["row_id_max"],
                "selection_start": int(selection_meta["start"]),
                "selection_stop": int(selection_meta["stop"]),
                "selection_shard_size": int(selection_meta["shard_size"]),
                "selection_mode": selection_meta["selection_mode"],
                "db_slice_seconds": float(db_slice_seconds),
                "lmdb_slice_seconds": float(lmdb_slice_seconds),
                "dm_infer_seconds": float(dm_infer_seconds),
                "lmdb_copied": int(lmdb_meta["copied"]),
                "lmdb_missing": int(len(lmdb_meta["missing"])),
                "lmdb_worker_count": int(lmdb_meta["worker_count"]),
                "lmdb_fallback_lookups": int(lmdb_meta["fallback_lookups"]),
                "successful_items": len(summary),
                "results_dir": str(results_dir),
            }
        )
    except Exception as exc:
        _log(f"task-failed error={repr(exc)}")
        payload.update(
            {
                "success": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        (results_dir / "task_result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

    (results_dir / "task_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    task_stdout_handle.flush()


if __name__ == "__main__":
    main()
