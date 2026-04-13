import gc
import heapq
import multiprocessing as mp
import os
import queue as pyqueue
import shutil
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

from ase.db import connect
from ase.io import write
from tqdm import tqdm

from emoles.build.uma_core import (
    _coerce_int,
    _derive_names,
    _extract_source_row_id_from_row,
    _get_row_data,
    _get_row_key_value_pairs,
    _infer_charge,
    _jsonify,
    _merge_row_metadata,
    _prepare_db_key_value_pairs,
    _safe_log,
    load_fairchem_calculator as _base_load_fairchem_calculator,
    sanitize_name,
)
from emoles.utils.parallel import (
    build_worker_gpu_plan as _build_worker_gpu_plan,
    configure_worker_env as _base_configure_worker_env,
    query_gpu_mem_mb as _query_gpu_mem_mb,
    resolve_gpu_ids as _resolve_gpu_ids,
    set_thread_env as _set_thread_env,
)

def _prepare_workspace(workspace: str, write_xyz: bool = False):
    os.makedirs(workspace, exist_ok=True)
    out_db_path = os.path.join(workspace, "optimized_all.db")
    fail_log_path = os.path.join(workspace, "failed_jobs.log")
    shard_db_dir = os.path.join(workspace, "worker_db_shards")
    out_xyz_dir = os.path.join(workspace, "optimized_xyz_all") if write_xyz else None

    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    if os.path.exists(fail_log_path):
        os.remove(fail_log_path)
    if os.path.exists(shard_db_dir):
        shutil.rmtree(shard_db_dir)
    os.makedirs(shard_db_dir, exist_ok=True)
    if write_xyz:
        if os.path.exists(out_xyz_dir):
            shutil.rmtree(out_xyz_dir)
        os.makedirs(out_xyz_dir, exist_ok=True)

    return out_xyz_dir, shard_db_dir, out_db_path, fail_log_path


def _prepare_model_assets(model_name: str, workspace: str):
    from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies

    _ = get_isolated_atomic_energies(model_name, workspace)


def _configure_worker_runtime(cpu_threads: Optional[int], gpu_id: Optional[int]):
    _base_configure_worker_env(
        gpu_id=gpu_id,
        cpu_threads_per_worker=cpu_threads,
    )


# ==========================================
# Parent-death protection
# ==========================================
def _best_effort_set_pdeathsig():
    if os.name != "posix":
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def _start_parent_watchdog(interval_sec: float = 5.0):
    parent_pid = os.getppid()

    def check_parent():
        while True:
            try:
                current_ppid = os.getppid()
                if current_ppid != parent_pid or current_ppid == 1:
                    os._exit(1)
            except Exception:
                os._exit(1)
            time.sleep(interval_sec)

    t = threading.Thread(target=check_parent, daemon=True)
    t.start()
    return t


def _load_fairchem_calculator(checkpoint_path: str, model_name: str, workspace: str, device: str):
    return _base_load_fairchem_calculator(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        workspace=workspace,
        device=device,
    )


# ==========================================
# Feeder
# ==========================================
def _task_feeder(input_db: str, task_queue, result_queue, stop_event):
    try:
        with connect(input_db) as src_db:
            for row in src_db.select():
                if stop_event.is_set():
                    break
                rid = int(row.id)
                while True:
                    if stop_event.is_set():
                        break
                    try:
                        task_queue.put(rid, timeout=1.0)
                        break
                    except pyqueue.Full:
                        continue
        result_queue.put({"type": "feeder_done"})
    except Exception as e:
        result_queue.put(
            {
                "type": "feeder_error",
                "error": f"Task feeder failed: {e}",
                "traceback": traceback.format_exc(),
            }
        )


# ==========================================
# Merge shard DBs -> final DB (dedup)
# ==========================================
def _merge_worker_dbs(
    worker_db_paths: List[str],
    out_db_path: str,
    write_xyz: bool = False,
    out_xyz_dir: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
    dedup_on_source_row_id: bool = True,
) -> int:
    existing_paths = [p for p in worker_db_paths if os.path.exists(p)]
    if os.path.exists(out_db_path):
        os.remove(out_db_path)

    if len(existing_paths) == 0:
        with connect(out_db_path):
            pass
        return 0

    shard_infos = []
    total_rows = 0
    try:
        for shard_idx, path in enumerate(existing_paths):
            db = connect(path)
            cnt = db.count()
            if cnt <= 0:
                try:
                    db.close()
                except Exception:
                    pass
                continue
            total_rows += cnt
            shard_infos.append(
                {
                    "shard_idx": shard_idx,
                    "path": path,
                    "db": db,
                    "iter": db.select(sort="source_row_id"),
                }
            )

        if total_rows == 0:
            with connect(out_db_path):
                pass
            return 0

        heap = []
        for info in shard_infos:
            try:
                row = next(info["iter"])
                src_id = _extract_source_row_id_from_row(row)
                heapq.heappush(heap, (src_id, info["shard_idx"], row, info["iter"]))
            except StopIteration:
                pass

        merged = 0
        dup = 0
        seen = set()

        with connect(out_db_path) as tgt_db:
            pbar = tqdm(
                total=total_rows,
                desc="Merging shards",
                unit="row",
                disable=not show_progress,
                dynamic_ncols=True,
            )
            try:
                while heap:
                    src_id, shard_idx, row, it = heapq.heappop(heap)

                    if dedup_on_source_row_id and src_id in seen:
                        dup += 1
                    else:
                        atoms = row.toatoms()
                        data = _get_row_data(row)
                        kvp = _get_row_key_value_pairs(row)

                        ch = _coerce_int(data.get("charge", None), None)
                        sp = _coerce_int(data.get("spin", None), None)
                        if ch is not None:
                            atoms.info["charge"] = int(ch)
                        if sp is not None:
                            atoms.info["spin"] = int(sp)

                        tgt_db.write(atoms, data=data, **kvp)

                        if write_xyz:
                            name = data.get("optimized_name", None) or data.get("base_name", None) or f"id_{src_id:06d}"
                            out_xyz = os.path.join(out_xyz_dir, f"{sanitize_name(name)}.xyz")
                            write(out_xyz, atoms)

                        merged += 1
                        if dedup_on_source_row_id:
                            seen.add(src_id)

                    pbar.update(1)

                    try:
                        next_row = next(it)
                        next_src_id = _extract_source_row_id_from_row(next_row)
                        heapq.heappush(heap, (next_src_id, shard_idx, next_row, it))
                    except StopIteration:
                        pass

                if verbose:
                    _safe_log(f"[MERGE] merged rows: {merged} (skipped dup: {dup})", show_progress)
            finally:
                pbar.close()

        return merged
    finally:
        for info in shard_infos:
            try:
                info["db"].close()
            except Exception:
                pass


# ==========================================
# Worker
# ==========================================
def _worker_loop(
    slot_id: int,
    attempt: int,
    gpu_id: Optional[int],
    input_db: str,
    worker_db_path: str,
    checkpoint_path: str,
    model_name: str,
    workspace: str,
    fmax: float,
    max_steps: int,
    cpu_threads_per_worker: Optional[int],
    task_queue,
    result_queue,
    stop_event,
    empty_cache_every: int = 0,
):
    _best_effort_set_pdeathsig()
    _start_parent_watchdog(interval_sec=5.0)

    pid = os.getpid()
    physical_device = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    fairchem_device = "cpu" if gpu_id is None else "cuda"
    input_db_abs = os.path.abspath(input_db)

    t_init0 = time.time()
    try:
        _configure_worker_runtime(cpu_threads_per_worker, gpu_id)
        calc = _load_fairchem_calculator(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            workspace=workspace,
            device=fairchem_device,
        )
        init_s = time.time() - t_init0

        mem_alloc_mb, mem_reserved_mb = None, None
        try:
            import torch

            if gpu_id is not None and torch.cuda.is_available():
                mem_alloc_mb = int(torch.cuda.memory_allocated() / 1024 / 1024)
                mem_reserved_mb = int(torch.cuda.memory_reserved() / 1024 / 1024)
        except Exception:
            pass

        result_queue.put(
            {
                "type": "worker_ready",
                "slot_id": slot_id,
                "attempt": attempt,
                "pid": pid,
                "device": physical_device,
                "worker_db": worker_db_path,
                "init_s": float(init_s),
                "mem_alloc_mb": mem_alloc_mb,
                "mem_reserved_mb": mem_reserved_mb,
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "type": "worker_fatal",
                "stage": "init",
                "slot_id": slot_id,
                "attempt": attempt,
                "pid": pid,
                "device": physical_device,
                "error": f"Model init failed: {e}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    task_i = 0
    try:
        with connect(input_db) as src_db, connect(worker_db_path) as out_db:
            while True:
                try:
                    row_id = task_queue.get(timeout=1.0)
                except pyqueue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if row_id is None:
                    break

                safe_name = f"id_{int(row_id):06d}"
                try:
                    row = src_db.get(id=int(row_id))
                    atoms = row.toatoms()

                    src_atoms_info = dict(getattr(atoms, "info", {}) or {})
                    src_kvp = _get_row_key_value_pairs(row)
                    src_data = _get_row_data(row)
                    merged_meta = dict(_merge_row_metadata(row, atoms))

                    charge = _infer_charge(atoms, merged_meta)
                    spin = 1
                    atoms.info["charge"] = int(charge)
                    atoms.info["spin"] = int(spin)

                    base_name, safe_name = _derive_names(row, merged_meta)

                    result_queue.put(
                        {
                            "type": "started",
                            "slot_id": slot_id,
                            "attempt": attempt,
                            "pid": pid,
                            "device": physical_device,
                            "row_id": int(row.id),
                            "name": safe_name,
                            "charge": int(charge),
                            "spin": int(spin),
                        }
                    )

                    atoms.calc = calc
                    opt = LBFGS(atoms, logfile=None)
                    converged = bool(opt.run(fmax=fmax, steps=max_steps))
                    atoms.calc = None
                    nsteps = getattr(opt, "nsteps", None)

                    record_data = {
                        "charge": int(charge),
                        "spin": int(spin),
                        "base_name": base_name,
                        "optimized_name": safe_name,
                        "source_row_id": int(row.id),
                        "source_unique_id": _jsonify(getattr(row, "unique_id", None)),
                        "source_db": input_db_abs,
                        "_source_key_value_pairs": _jsonify(src_kvp),
                        "_source_data": _jsonify(src_data),
                        "_source_atoms_info": _jsonify(src_atoms_info),
                        "_runtime": _jsonify(
                            {
                                "device": physical_device,
                                "slot_id": int(slot_id),
                                "attempt": int(attempt),
                                "pid": int(pid),
                                "worker_db": os.path.basename(worker_db_path),
                            }
                        ),
                        "_optimization": _jsonify(
                            {
                                "optimizer": "LBFGS",
                                "fmax": float(fmax),
                                "max_steps": int(max_steps),
                                "nsteps": _jsonify(nsteps),
                                "converged": bool(converged),
                            }
                        ),
                    }

                    searchable_meta = dict(merged_meta)
                    searchable_meta.update(
                        {
                            "source_row_id": int(row.id),
                            "source_unique_id": str(getattr(row, "unique_id", "")),
                            "base_name": base_name,
                            "optimized_name": safe_name,
                            "source_db_name": os.path.basename(input_db_abs),
                            "opt_converged": bool(converged),
                            "slot_id": int(slot_id),
                            "attempt": int(attempt),
                            "pid": int(pid),
                            "worker_db": os.path.basename(worker_db_path),
                            "device_name": physical_device,
                        }
                    )
                    db_kvp = _prepare_db_key_value_pairs(searchable_meta)

                    out_db.write(atoms, data=record_data, **db_kvp)

                    result_queue.put(
                        {
                            "type": "done",
                            "slot_id": slot_id,
                            "attempt": attempt,
                            "pid": pid,
                            "device": physical_device,
                            "row_id": int(row.id),
                            "name": safe_name,
                            "charge": int(charge),
                            "spin": int(spin),
                            "converged": bool(converged),
                        }
                    )

                except Exception as e:
                    result_queue.put(
                        {
                            "type": "error",
                            "slot_id": slot_id,
                            "attempt": attempt,
                            "pid": pid,
                            "device": physical_device,
                            "row_id": int(row_id),
                            "name": safe_name,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                finally:
                    task_i += 1
                    if empty_cache_every and (task_i % int(empty_cache_every) == 0):
                        try:
                            import torch

                            if gpu_id is not None and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        gc.collect()

    except Exception as e:
        result_queue.put(
            {
                "type": "worker_fatal",
                "stage": "run",
                "slot_id": slot_id,
                "attempt": attempt,
                "pid": pid,
                "device": physical_device,
                "error": f"Worker crashed: {e}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            import torch

            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            result_queue.put(
                {
                    "type": "worker_exit",
                    "slot_id": slot_id,
                    "attempt": attempt,
                    "pid": pid,
                    "device": physical_device,
                }
            )
        except Exception:
            pass


# ==========================================
# Main Entry
# ==========================================
