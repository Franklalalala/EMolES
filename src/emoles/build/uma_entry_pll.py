# uma_entry.py
import os
import re
import shutil
import argparse
import traceback
import queue as pyqueue
import multiprocessing as mp
import threading
import time
import heapq
import gc
import subprocess
from typing import Optional, List, Any, Dict, Tuple

import numpy as np
from ase import Atoms
from ase.io import write
from ase.db import connect
from ase.db.core import check as ase_db_check
from ase.optimize import LBFGS
from tqdm import tqdm

# Optional import for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ==========================================
# Default Configuration
# ==========================================
DEFAULT_CHECKPOINT = r"/home/user/openequi_workspace/my_run/checkpoints/uma-m-1p1.pt"
DEFAULT_WORKSPACE = os.path.abspath("out_li_clusters")
DEFAULT_MODEL_NAME = "uma-m-1p1"


# ==========================================
# Utility
# ==========================================
def sanitize_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unnamed"


def _coerce_int(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return int(float(x))
        return int(float(x))
    except Exception:
        return default


def _normalize_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _is_db_scalar(v) -> bool:
    v = _normalize_scalar(v)
    return isinstance(v, (str, int, float, bool))


def _jsonify(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return repr(obj)


def _prepare_db_key_value_pairs(meta: dict) -> dict:
    kvp = {}
    for k, v in (meta or {}).items():
        if not isinstance(k, str):
            continue
        v = _normalize_scalar(v)
        if v is None or (not _is_db_scalar(v)):
            continue
        try:
            ase_db_check({k: v})
            kvp[k] = v
        except Exception:
            continue
    return kvp


def _safe_log(msg: str, show_progress: bool = True):
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg)


def _derive_names(row, meta: dict):
    raw_name = meta.get("xyz_file", None) or meta.get("name", None) or f"id_{row.id}"
    raw_name = os.path.splitext(str(raw_name))[0]
    base_name = sanitize_name(raw_name)
    unique_name = f"id_{int(row.id):06d}__{base_name}"
    return base_name, unique_name


# ==========================================
# SMILES / Input DB
# ==========================================
def smiles_to_atoms(smiles: str) -> Atoms:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to process SMILES strings.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res == -1:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if res == -1:
        raise RuntimeError(f"RDKit failed to embed SMILES: {smiles}")
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def prepare_input_source(workspace: str, input_db: str = None, smiles_list: list = None) -> str:
    os.makedirs(workspace, exist_ok=True)
    if smiles_list:
        temp_db_path = os.path.join(workspace, "temp_smiles_input.db")
        print(f"Generating 3D structures from {len(smiles_list)} SMILES -> {temp_db_path} ...")
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        with connect(temp_db_path) as db:
            for i, smi in enumerate(smiles_list):
                try:
                    atoms = smiles_to_atoms(smi)
                    name = f"smiles_{i}_{sanitize_name(smi[:10])}"
                    db.write(atoms, name=name, smiles=smi)
                except Exception as e:
                    print(f"[Warning] Failed to convert SMILES '{smi}': {e}")
        return temp_db_path

    if not input_db:
        return os.path.join(workspace, "all.db")
    return input_db


# ==========================================
# Metadata / Charge
# ==========================================
def _get_row_key_value_pairs(row) -> dict:
    try:
        kvp = getattr(row, "key_value_pairs", None)
        return dict(kvp) if kvp else {}
    except Exception:
        return {}


def _get_row_data(row) -> dict:
    try:
        data = getattr(row, "data", None)
        return dict(data) if data else {}
    except Exception:
        return {}


def _merge_row_metadata(row, atoms: Atoms) -> dict:
    meta = {}
    meta.update(_get_row_key_value_pairs(row))
    meta.update(_get_row_data(row))
    try:
        if isinstance(getattr(atoms, "info", None), dict):
            for k, v in atoms.info.items():
                if k not in meta:
                    meta[k] = v
    except Exception:
        pass
    return meta


def _infer_charge(atoms: Atoms, meta: dict) -> int:
    if "charge" in meta and meta["charge"] is not None:
        ch = _coerce_int(meta["charge"], default=None)
        if ch is not None:
            return int(ch)

    n_anion = None
    for key in ("n_anion", "n_anion_total", "n_anions", "n_anion_tot"):
        if key in meta and meta[key] is not None:
            n_anion = _coerce_int(meta[key], default=None)
            if n_anion is not None:
                break

    has_li = ("Li" in atoms.get_chemical_symbols())
    if n_anion is not None:
        return int((1 - n_anion) if has_li else (0 - n_anion))
    return int(1 if has_li else 0)


def _extract_source_row_id_from_row(row) -> int:
    kvp = _get_row_key_value_pairs(row)
    if "source_row_id" in kvp:
        x = _coerce_int(kvp["source_row_id"], None)
        if x is not None:
            return int(x)
    data = _get_row_data(row)
    if "source_row_id" in data:
        x = _coerce_int(data["source_row_id"], None)
        if x is not None:
            return int(x)
    raise KeyError("source_row_id not found in shard row")


# ==========================================
# Workspace / Device Planning
# ==========================================
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


def _resolve_gpu_ids(device: str, gpus: Optional[List[str]]) -> List[int]:
    dev = str(device).lower().strip()
    if dev == "cpu":
        return []
    if gpus:
        return [int(x) for x in gpus]
    if dev.startswith("cuda:"):
        return [int(dev.split(":")[1])]
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n <= 0:
        raise RuntimeError("No CUDA GPU detected, but device is not cpu.")
    return list(range(n))


def _build_worker_gpu_plan(device: str, gpus: Optional[List[str]], workers_per_gpu: int, cpu_workers: Optional[int]):
    dev = str(device).lower().strip()
    if dev == "cpu":
        n = cpu_workers or 1
        return [None] * max(1, int(n))

    gpu_ids = _resolve_gpu_ids(device, gpus)
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be >= 1")

    plan = []
    for gid in gpu_ids:
        plan.extend([gid] * int(workers_per_gpu))
    return plan


def _set_thread_env(cpu_threads_per_worker: Optional[int]):
    if cpu_threads_per_worker is None or cpu_threads_per_worker <= 0:
        return
    val = str(int(cpu_threads_per_worker))
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[k] = val


def _prepare_model_assets(model_name: str, workspace: str):
    from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies

    _ = get_isolated_atomic_energies(model_name, workspace)


# ==========================================
# GPU memory query (for ramp-up decisions)
# ==========================================
def _query_gpu_mem_mb(gpu_id: int) -> Optional[Tuple[int, int]]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(info.used // (1024 * 1024)), int(info.total // (1024 * 1024))
    except Exception:
        pass

    try:
        cmd = [
            "nvidia-smi",
            "-i",
            str(int(gpu_id)),
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        parts = [x.strip() for x in out.split(",")]
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None
    return None


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


# ==========================================
# Worker-side Runtime / FAIRChem loading
# ==========================================
def _configure_worker_runtime(cpu_threads: Optional[int], gpu_id: Optional[int]):
    _set_thread_env(cpu_threads)

    # MUST set before importing torch
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch  # noqa: F401

    try:
        if cpu_threads is not None and cpu_threads > 0:
            try:
                torch.set_num_threads(int(cpu_threads))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if gpu_id is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)  # visible idx
            except Exception:
                pass
    except Exception:
        pass


def _load_fairchem_calculator(checkpoint_path: str, model_name: str, workspace: str, device: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies

    atom_refs = get_isolated_atomic_energies(model_name, workspace)
    predictor = load_predict_unit(checkpoint_path, "default", None, device, atom_refs)
    calc = FAIRChemCalculator(predictor, task_name="omol")
    return calc


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
def entry(
    input_db: str = None,
    smiles: list = None,
    workspace: str = DEFAULT_WORKSPACE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    gpus: Optional[List[str]] = None,
    workers_per_gpu: int = 4,
    cpu_workers: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    verbose: bool = False,
    show_progress: bool = True,
    queue_size: Optional[int] = None,
    write_xyz: bool = False,
    keep_worker_dbs: bool = False,
    # ramp-up
    warmup_workers_per_gpu: int = 2,
    ramp_step_per_gpu: int = 1,
    ramp_interval_sec: float = 60.0,
    stable_window_sec: float = 60.0,
    gpu_headroom_gb: float = 13.0,
    # init control
    init_timeout_sec: float = 900.0,
    startup_stagger_sec: float = 0.2,  # spawn pacing, non-blocking
    global_init_concurrency: int = 1,
    init_concurrency_per_gpu: int = 1,
    # respawn
    max_restarts_per_slot: int = 200,
    restart_backoff_sec: float = 2.0,
    empty_cache_every: int = 0,
    pytorch_alloc_conf: Optional[str] = None,
    task_timeout_sec: Optional[float] = None,
) -> str:
    os.makedirs(workspace, exist_ok=True)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_alloc_conf)

    out_xyz_dir, shard_db_dir, out_db_path, fail_log_path = _prepare_workspace(workspace, write_xyz=write_xyz)
    active_input_db = prepare_input_source(workspace, input_db, smiles)
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db:
        total = src_db.count()
    if total == 0:
        with connect(out_db_path):
            pass
        return out_db_path

    worker_gpu_plan = _build_worker_gpu_plan(device=device, gpus=gpus, workers_per_gpu=workers_per_gpu, cpu_workers=cpu_workers)
    total_slots = len(worker_gpu_plan)
    if total_slots <= 0:
        raise RuntimeError("No workers planned.")

    gpu_ids = sorted(set([x for x in worker_gpu_plan if x is not None]))
    if str(device).lower() == "cpu":
        gpu_ids = []

    # CPU threads default: average distribution across all planned slots
    if cpu_threads_per_worker is None:
        ncpu = os.cpu_count() or 1
        cpu_threads_per_worker = max(1, ncpu // max(1, total_slots))
    _set_thread_env(cpu_threads_per_worker)

    if device.lower() != "cpu":
        _prepare_model_assets(model_name, workspace)

    # slots_by_gpu + rank map (avoid list.index in hot path)
    slots_by_gpu: Dict[Optional[int], List[int]] = {}
    slot_rank_in_gpu: Dict[int, int] = {}
    for sid, gid in enumerate(worker_gpu_plan):
        slots_by_gpu.setdefault(gid, []).append(sid)
    for gid, sids in slots_by_gpu.items():
        if gid is None:
            continue
        for r, sid in enumerate(sids):
            slot_rank_in_gpu[sid] = r

    warmup = int(max(0, min(warmup_workers_per_gpu, workers_per_gpu)))
    if str(device).lower() == "cpu":
        warmup = min(warmup_workers_per_gpu, total_slots) if warmup_workers_per_gpu > 0 else total_slots

    ctx = mp.get_context("spawn")
    task_qsize = max(64, total_slots * 4) if queue_size is None else int(queue_size)
    task_queue = ctx.Queue(maxsize=task_qsize)
    result_queue = ctx.Queue()
    stop_event = ctx.Event()

    feeder_thread = threading.Thread(target=_task_feeder, args=(active_input_db, task_queue, result_queue, stop_event), daemon=True)
    feeder_thread.start()

    # slot state
    slot_proc: List[Optional[mp.Process]] = [None] * total_slots
    slot_pid: List[Optional[int]] = [None] * total_slots
    slot_attempt: List[int] = [0] * total_slots
    slot_spawn_ts: List[float] = [0.0] * total_slots
    slot_ready: List[bool] = [False] * total_slots
    slot_next_spawn_ts: List[float] = [0.0] * total_slots
    slot_restarts: List[int] = [0] * total_slots

    inflight: Dict[int, Tuple[int, float, str]] = {}
    finished, ok_set, fail_set = set(), set(), set()
    running: Dict[str, int] = {}
    all_shard_db_paths: List[str] = []

    current_target_per_gpu = warmup if warmup > 0 else (1 if workers_per_gpu > 0 else 0)
    current_target_per_gpu = int(min(current_target_per_gpu, workers_per_gpu))
    if str(device).lower() == "cpu":
        current_target_per_gpu = total_slots

    last_failure_ts = 0.0
    last_ramp_ts = 0.0
    pending_spawn: List[int] = []
    global_next_spawn_ts = 0.0  # non-blocking spawn pacing

    def _dev_name(gid: Optional[int]) -> str:
        return "cpu" if gid is None else f"cuda:{gid}"

    def _is_slot_active(slot_id: int) -> bool:
        gid = worker_gpu_plan[slot_id]
        if str(device).lower() == "cpu":
            return True
        if gid is None:
            return False
        r = slot_rank_in_gpu.get(slot_id, 10**9)
        return r < current_target_per_gpu

    def _active_slots() -> List[int]:
        return [sid for sid in range(total_slots) if _is_slot_active(sid)]

    def _gpu_mem_ok_for_ramp() -> Tuple[bool, Dict[int, Tuple[int, int]]]:
        stats: Dict[int, Tuple[int, int]] = {}
        if str(device).lower() == "cpu":
            return True, stats
        headroom_mb = int(float(gpu_headroom_gb) * 1024)
        for gid in gpu_ids:
            mem = _query_gpu_mem_mb(gid)
            if mem is None:
                return False, stats
            used, total_mb = mem
            stats[gid] = (used, total_mb)
            if used >= (total_mb - headroom_mb):
                return False, stats
        return True, stats

    def _retire_process(slot_id: int, why: str):
        """
        Make best-effort to terminate/join old process to avoid zombies/handles leak.
        Never blocks long.
        """
        p = slot_proc[slot_id]
        if p is None:
            return
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
        try:
            p.join(timeout=0.2)
        except Exception:
            pass

    def _log_fail(line: str):
        fail_fp.write(line + "\n")
        fail_fp.flush()
        _safe_log(line, show_progress)

    def _schedule_respawn(slot_id: int, why: str, exitcode: Optional[int] = None):
        nonlocal last_failure_ts
        last_failure_ts = time.time()

        # requeue inflight if any
        if slot_id in inflight:
            rid, st, dev = inflight.pop(slot_id)
            running[dev] = max(0, running.get(dev, 0) - 1)
            # best-effort requeue
            while True:
                if stop_event.is_set():
                    break
                try:
                    task_queue.put(int(rid), timeout=1.0)
                    break
                except pyqueue.Full:
                    continue

        slot_ready[slot_id] = False

        slot_restarts[slot_id] += 1
        slot_attempt[slot_id] += 1

        n = slot_restarts[slot_id]
        backoff = float(restart_backoff_sec) * (2 ** min(n, 6))
        slot_next_spawn_ts[slot_id] = time.time() + min(backoff, 120.0)

        dev = _dev_name(worker_gpu_plan[slot_id])
        _log_fail(
            f"[RESPAWN][{dev}][S{slot_id:03d}] restarts={n} next_in={slot_next_spawn_ts[slot_id]-time.time():.1f}s why={why} exit={exitcode}"
        )

    def _count_global_inits() -> int:
        # include only active slots
        c = 0
        for sid in _active_slots():
            p = slot_proc[sid]
            if p is not None and (not slot_ready[sid]):
                # if dead but not cleaned yet, treat as not occupying by checking is_alive
                if p.is_alive():
                    c += 1
        return c

    def _count_gpu_inits(gid: int) -> int:
        c = 0
        for sid in slots_by_gpu.get(gid, []):
            if not _is_slot_active(sid):
                continue
            p = slot_proc[sid]
            if p is not None and (not slot_ready[sid]) and p.is_alive():
                c += 1
        return c

    def _can_spawn(slot_id: int) -> bool:
        # main-process "token bucket" without locks
        if global_init_concurrency > 0:
            if _count_global_inits() >= int(global_init_concurrency):
                return False

        gid = worker_gpu_plan[slot_id]
        if gid is not None and init_concurrency_per_gpu > 0:
            if _count_gpu_inits(int(gid)) >= int(init_concurrency_per_gpu):
                return False

        return True

    def _spawn_slot(slot_id: int, reason: str = "") -> bool:
        if stop_event.is_set():
            return False
        if len(finished) >= total:
            return False
        if not _is_slot_active(slot_id):
            return False
        if slot_restarts[slot_id] >= int(max_restarts_per_slot):
            return False
        if time.time() < slot_next_spawn_ts[slot_id]:
            return False

        p = slot_proc[slot_id]
        if p is not None and p.is_alive():
            return False

        gid = worker_gpu_plan[slot_id]
        attempt = slot_attempt[slot_id]
        worker_db_path = os.path.join(shard_db_dir, f"slot_{slot_id:03d}__a{attempt:04d}.db")
        all_shard_db_paths.append(worker_db_path)

        try:
            p = ctx.Process(
                target=_worker_loop,
                args=(
                    slot_id,
                    attempt,
                    gid,
                    active_input_db,
                    worker_db_path,
                    checkpoint_path,
                    model_name,
                    workspace,
                    fmax,
                    max_steps,
                    cpu_threads_per_worker,
                    task_queue,
                    result_queue,
                    stop_event,
                    int(empty_cache_every),
                ),
            )
            p.start()
        except Exception as e:
            # spawn failure -> schedule respawn
            slot_proc[slot_id] = None
            slot_pid[slot_id] = None
            slot_ready[slot_id] = False
            _schedule_respawn(slot_id, why=f"spawn_failed: {e}", exitcode=None)
            return False

        slot_proc[slot_id] = p
        slot_pid[slot_id] = p.pid
        slot_spawn_ts[slot_id] = time.time()
        slot_ready[slot_id] = False

        if verbose:
            _safe_log(
                f"[SPAWN][{_dev_name(gid)}][S{slot_id:03d}] attempt={attempt} pid={p.pid} reason={reason}",
                show_progress,
            )
        return True

    def _monitor_slots():
        # high priority: clean dead procs quickly to not "occupy" init tokens
        now = time.time()
        for sid in _active_slots():
            p = slot_proc[sid]

            # dead
            if p is not None and (not p.is_alive()):
                exitcode = p.exitcode
                try:
                    p.join(timeout=0.0)
                except Exception:
                    pass
                slot_proc[sid] = None
                slot_pid[sid] = None
                _schedule_respawn(sid, why="proc_dead", exitcode=exitcode)
                continue

            # init timeout
            if p is not None and (not slot_ready[sid]) and init_timeout_sec:
                if (now - slot_spawn_ts[sid]) > float(init_timeout_sec):
                    dev = _dev_name(worker_gpu_plan[sid])
                    _log_fail(f"[KILL ][{dev}][S{sid:03d}] init_timeout t={now-slot_spawn_ts[sid]:.1f}s")
                    _retire_process(sid, why="init_timeout")
                    slot_proc[sid] = None
                    slot_pid[sid] = None
                    _schedule_respawn(sid, why="init_timeout", exitcode=None)
                    continue

            # add to pending queue if missing
            if slot_proc[sid] is None and sid not in pending_spawn:
                pending_spawn.append(sid)

    def _maybe_ramp_up():
        nonlocal current_target_per_gpu, last_ramp_ts
        if str(device).lower() == "cpu":
            return
        if current_target_per_gpu >= int(workers_per_gpu):
            return

        now = time.time()
        if now - last_ramp_ts < float(ramp_interval_sec):
            return
        if last_failure_ts and (now - last_failure_ts) < float(stable_window_sec):
            return

        # require all current active slots are alive+ready
        for sid in _active_slots():
            p = slot_proc[sid]
            if p is None or (not p.is_alive()) or (not slot_ready[sid]):
                return

        ok_mem, stats = _gpu_mem_ok_for_ramp()
        if not ok_mem:
            if verbose and stats:
                s = " ".join([f"gpu{gid}:{u}/{t}MB" for gid, (u, t) in stats.items()])
                _safe_log(f"[RAMP ] skip (mem headroom) | {s}", show_progress)
            return

        current_target_per_gpu = min(int(workers_per_gpu), int(current_target_per_gpu) + int(ramp_step_per_gpu))
        last_ramp_ts = now

        s = ""
        if stats:
            s = " | " + " ".join([f"gpu{gid}:{u}/{t}MB" for gid, (u, t) in stats.items()])
        _safe_log(f"[RAMP ] target_workers_per_gpu -> {current_target_per_gpu}{s}", show_progress)

        # enqueue newly activated slots
        for gid in gpu_ids:
            for sid in slots_by_gpu.get(gid, []):
                if _is_slot_active(sid) and slot_proc[sid] is None and sid not in pending_spawn:
                    pending_spawn.append(sid)

    def _is_stale_msg(msg: dict) -> bool:
        sid = msg.get("slot_id", None)
        if sid is None or (not (0 <= int(sid) < total_slots)):
            return True
        sid = int(sid)
        if msg.get("pid", None) != slot_pid[sid]:
            return True
        if msg.get("attempt", None) != slot_attempt[sid]:
            return True
        return False

    # seed pending queue for initial warmup
    for sid in _active_slots():
        pending_spawn.append(sid)

    feeder_done = False
    normal_shutdown = False

    if verbose:
        _safe_log(
            f"[INIT ] gpus={gpu_ids} warmup={warmup} target={current_target_per_gpu}/{workers_per_gpu} "
            f"global_init_conc={global_init_concurrency} per_gpu_init_conc={init_concurrency_per_gpu} "
            f"cpu_threads_per_worker={cpu_threads_per_worker}",
            show_progress,
        )

    with open(fail_log_path, "w", encoding="utf-8") as fail_fp:
        pbar = tqdm(total=total, desc="Optimizing", unit="mol", disable=not show_progress, dynamic_ncols=True)
        last_monitor = 0.0

        def _refresh_postfix():
            if not show_progress:
                return
            alive = sum(1 for sid in _active_slots() if slot_proc[sid] is not None and slot_proc[sid].is_alive())
            active_running = sum(running.values()) if running else 0
            init_global = _count_global_inits()
            postfix = {
                "ok": len(ok_set),
                "fail": len(fail_set),
                "run": active_running,
                "init": init_global,
                "alive": alive,
                "tgt": current_target_per_gpu,
            }
            gpu_items = [f"{k.split(':')[-1]}:{running[k]}" for k in sorted(running.keys())]
            if gpu_items:
                postfix["g"] = ",".join(gpu_items)
            pbar.set_postfix(postfix)

        try:
            while len(finished) < total:
                now = time.time()

                # monitor 1Hz
                if now - last_monitor > 1.0:
                    last_monitor = now

                    # task timeout kill
                    if task_timeout_sec is not None:
                        timeout = float(task_timeout_sec)
                        for sid, (rid, st, dev) in list(inflight.items()):
                            if (now - st) > timeout:
                                p = slot_proc[sid]
                                devn = _dev_name(worker_gpu_plan[sid])
                                _log_fail(f"[KILL ][{devn}][S{sid:03d}] task_timeout row={rid} t={now-st:.1f}s")
                                _retire_process(sid, why="task_timeout")
                                slot_proc[sid] = None
                                slot_pid[sid] = None
                                _schedule_respawn(sid, why=f"task_timeout row={rid}", exitcode=None)

                    _monitor_slots()
                    _maybe_ramp_up()
                    _refresh_postfix()

                # non-blocking spawn pacing: at most 1 spawn per tick, and only if tokens available
                if pending_spawn and now >= global_next_spawn_ts:
                    # try find one spawnable slot
                    spawned = False
                    for i, sid in enumerate(list(pending_spawn)):
                        if time.time() < slot_next_spawn_ts[sid]:
                            continue
                        if not _can_spawn(sid):
                            continue
                        pending_spawn.pop(i)
                        if _spawn_slot(sid, reason="queue"):
                            global_next_spawn_ts = time.time() + float(startup_stagger_sec)
                            spawned = True
                        break
                    if spawned:
                        pass

                # consume result queue frequently
                try:
                    msg = result_queue.get(timeout=0.1)
                except pyqueue.Empty:
                    continue

                mtype = msg.get("type")

                if mtype in ("worker_ready", "started", "done", "error", "worker_fatal", "worker_exit"):
                    if _is_stale_msg(msg):
                        continue

                if mtype == "feeder_done":
                    feeder_done = True

                elif mtype == "feeder_error":
                    stop_event.set()
                    raise RuntimeError(f"[FATAL][FEEDER] {msg['error']}")

                elif mtype == "worker_ready":
                    sid = int(msg["slot_id"])
                    slot_ready[sid] = True
                    if verbose:
                        _safe_log(
                            f"[READY][{msg['device']}][S{sid:03d}] init_s={msg.get('init_s'):.1f} "
                            f"VRAM_alloc={msg.get('mem_alloc_mb')}MB resv={msg.get('mem_reserved_mb')}MB",
                            show_progress,
                        )

                elif mtype == "started":
                    sid, dev, rid = int(msg["slot_id"]), msg["device"], int(msg["row_id"])
                    running[dev] = running.get(dev, 0) + 1
                    inflight[sid] = (rid, time.time(), dev)
                    _refresh_postfix()

                elif mtype == "done":
                    sid, dev, rid = int(msg["slot_id"]), msg["device"], int(msg["row_id"])
                    inflight.pop(sid, None)
                    running[dev] = max(0, running.get(dev, 0) - 1)

                    if rid not in finished:
                        finished.add(rid)
                        ok_set.add(rid)
                        pbar.update(1)
                    _refresh_postfix()

                elif mtype == "error":
                    sid, dev, rid = int(msg["slot_id"]), msg["device"], int(msg.get("row_id", -1))
                    inflight.pop(sid, None)
                    running[dev] = max(0, running.get(dev, 0) - 1)

                    fail_fp.write(f"[ERROR][{dev}][S{sid:03d}] row={rid} | {msg['error']}\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()

                    if rid not in finished:
                        finished.add(rid)
                        fail_set.add(rid)
                        pbar.update(1)
                    _refresh_postfix()

                elif mtype == "worker_fatal":
                    sid = int(msg.get("slot_id", -1))
                    stage = msg.get("stage", "unknown")
                    devn = msg.get("device", "unknown")
                    fail_fp.write(f"[FATAL][{devn}][S{sid:03d}] stage={stage} | {msg['error']}\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()

                    # if still active, retire and respawn
                    if 0 <= sid < total_slots and _is_slot_active(sid):
                        _retire_process(sid, why="worker_fatal")
                        slot_proc[sid] = None
                        slot_pid[sid] = None
                        _schedule_respawn(sid, why=f"worker_fatal stage={stage}", exitcode=None)

                elif mtype == "worker_exit":
                    # normal exit; monitor will respawn if needed
                    pass

            normal_shutdown = True

        except KeyboardInterrupt:
            _safe_log("\n[Main] KeyboardInterrupt received, terminating workers...", show_progress)
            stop_event.set()
            raise
        finally:
            pbar.close()

    # ======================================
    # Shutdown
    # ======================================
    stop_event.set()

    # drain result queue briefly
    end_drain = time.time() + 1.0
    while time.time() < end_drain:
        try:
            result_queue.get(timeout=0.1)
        except pyqueue.Empty:
            pass

    # unblock workers
    for _ in range(total_slots * 2):
        try:
            task_queue.put_nowait(None)
        except Exception:
            break

    try:
        feeder_thread.join(timeout=3)
    except Exception:
        pass

    # join/terminate
    for sid in range(total_slots):
        p = slot_proc[sid]
        if p is None:
            continue
        try:
            p.join(timeout=3)
        except Exception:
            pass

    for sid in range(total_slots):
        p = slot_proc[sid]
        if p is not None and p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.join(timeout=1)
            except Exception:
                pass

    try:
        task_queue.cancel_join_thread()
    except Exception:
        pass
    try:
        result_queue.cancel_join_thread()
    except Exception:
        pass
    try:
        task_queue.close()
    except Exception:
        pass
    try:
        result_queue.close()
    except Exception:
        pass

    # ======================================
    # Merge shard DBs -> final DB
    # ======================================
    merged_rows = 0
    if normal_shutdown:
        _safe_log("[Main] Merging worker shard DBs into final output DB...", show_progress)
        merged_rows = _merge_worker_dbs(
            worker_db_paths=all_shard_db_paths,
            out_db_path=out_db_path,
            write_xyz=write_xyz,
            out_xyz_dir=out_xyz_dir,
            verbose=verbose,
            show_progress=show_progress,
            dedup_on_source_row_id=True,
        )
        if (not keep_worker_dbs) and os.path.exists(shard_db_dir):
            try:
                shutil.rmtree(shard_db_dir)
            except Exception:
                pass

    _safe_log(
        f"[Summary] total={total}, ok={len(ok_set)}, fail={len(fail_set)}, merged={merged_rows}, feeder_done={feeder_done}, output_db={out_db_path}",
        show_progress,
    )
    if len(fail_set) > 0:
        _safe_log(f"[Summary] failed log: {fail_log_path}", show_progress)

    return out_db_path


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="FAIRChem Parallel Optimization (Main-process state machine init throttling)")

    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Root output directory")
    parser.add_argument("--input-db", default=None, help="Input ASE database")
    parser.add_argument("--smiles", nargs="*", default=None, help="Input SMILES string(s) to optimize")

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name for atom refs cache")

    parser.add_argument("--device", default="cuda", help="Compute device: cuda / cuda:0 / cpu")
    parser.add_argument("--gpus", nargs="*", default=None, help="Physical GPU ids, e.g. --gpus 0 1")
    parser.add_argument("--workers-per-gpu", type=int, default=4, help="Max concurrent worker processes per GPU")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Only used when --device cpu")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=None, help="CPU threads per worker (default: avg split)")
    parser.add_argument("--queue-size", type=int, default=None, help="Task queue size (default: auto)")

    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence criteria")
    parser.add_argument("--steps", type=int, default=200, dest="max_steps", help="Max optimization steps")

    parser.add_argument("--write-xyz", action="store_true", help="Also write optimized xyz files")
    parser.add_argument("--keep-worker-dbs", action="store_true", help="Keep per-worker shard DBs after merge")

    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bar")
    parser.set_defaults(progress=True)

    # ramp-up
    parser.add_argument("--warmup-workers-per-gpu", type=int, default=2, help="Initial workers per GPU to start")
    parser.add_argument("--ramp-step-per-gpu", type=int, default=1, help="Workers per GPU added per ramp step")
    parser.add_argument("--ramp-interval-sec", type=float, default=60.0, help="Try ramp every N seconds")
    parser.add_argument("--stable-window-sec", type=float, default=60.0, help="Require no failures in last N seconds before ramp")
    parser.add_argument("--gpu-headroom-gb", type=float, default=13.0, help="Keep GPU headroom when ramping (GB)")

    # init control
    parser.add_argument("--init-timeout-sec", type=float, default=900.0, help="Kill+respawn if init not ready after N seconds")
    parser.add_argument("--startup-stagger-sec", type=float, default=0.2, help="Non-blocking spawn pacing (seconds)")
    parser.add_argument("--global-init-concurrency", type=int, default=1, help="Max concurrent inits across all GPUs")
    parser.add_argument("--init-concurrency-per-gpu", type=int, default=1, help="Max concurrent inits per GPU")

    # respawn
    parser.add_argument("--max-restarts-per-slot", type=int, default=200, help="Max respawns per slot")
    parser.add_argument("--restart-backoff-sec", type=float, default=2.0, help="Base backoff (exponential) for respawn")
    parser.add_argument("--empty-cache-every", type=int, default=0, help="torch.cuda.empty_cache every N tasks (0 disable)")
    parser.add_argument("--pytorch-alloc-conf", type=str, default=None, help="Set PYTORCH_CUDA_ALLOC_CONF")
    parser.add_argument("--task-timeout-sec", type=float, default=None, help="Kill+respawn if a task exceeds N seconds")

    args = parser.parse_args()

    out_path = entry(
        input_db=args.input_db,
        smiles=args.smiles,
        workspace=args.workspace,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        cpu_workers=args.cpu_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        fmax=args.fmax,
        max_steps=args.max_steps,
        verbose=args.verbose,
        show_progress=args.progress,
        queue_size=args.queue_size,
        write_xyz=args.write_xyz,
        keep_worker_dbs=args.keep_worker_dbs,
        warmup_workers_per_gpu=args.warmup_workers_per_gpu,
        ramp_step_per_gpu=args.ramp_step_per_gpu,
        ramp_interval_sec=args.ramp_interval_sec,
        stable_window_sec=args.stable_window_sec,
        gpu_headroom_gb=args.gpu_headroom_gb,
        init_timeout_sec=args.init_timeout_sec,
        startup_stagger_sec=args.startup_stagger_sec,
        global_init_concurrency=args.global_init_concurrency,
        init_concurrency_per_gpu=args.init_concurrency_per_gpu,
        max_restarts_per_slot=args.max_restarts_per_slot,
        restart_backoff_sec=args.restart_backoff_sec,
        empty_cache_every=args.empty_cache_every,
        pytorch_alloc_conf=args.pytorch_alloc_conf,
        task_timeout_sec=args.task_timeout_sec,
    )

    print(f"\nOptimization finished. Output DB: {out_path}")

# nohup sh -c 'python uma_entry.py   --input-db dataset_p2_active_cho.db   --workspace out_cho   --device cuda --gpus 0 1 2 3   --workers-per-gpu 14   --warmup-workers-per-gpu 2   --ramp-step-per-gpu 1   --ramp-interval-sec 60   --stable-window-sec 60   --global-init-concurrency 1   --init-concurrency-per-gpu 1   --gpu-headroom-gb 14   --startup-stagger-sec 0.2   --init-timeout-sec 1000' > run_cho.log 2>&1 &
if __name__ == "__main__":
    mp.freeze_support()
    main()