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
from typing import Optional, List, Any

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
DEFAULT_CHECKPOINT = r'/home/mingkang_nt/hetero_atoms_workspace/checkpoints/uma-m-1p1.pt'
DEFAULT_WORKSPACE = os.path.abspath('out_li_clusters')
DEFAULT_MODEL_NAME = "uma-m-1p1"


# ==========================================
# Utility
# ==========================================
def sanitize_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unnamed"


def _coerce_int(x, default=None):
    """Best-effort conversion to int (handles int/float/np scalars/strings)."""
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
    """Convert to JSON/ASE-data-friendly nested structure."""
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
    """
    只保留 ASE DB 允许的简单标量 key-value pairs。
    其他全部进入 data。
    """
    kvp = {}
    for k, v in (meta or {}).items():
        if not isinstance(k, str):
            continue
        v = _normalize_scalar(v)
        if v is None:
            continue
        if not _is_db_scalar(v):
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
    raw_name = meta.get('xyz_file', None) or meta.get('name', None) or f"id_{row.id}"
    raw_name = os.path.splitext(str(raw_name))[0]
    base_name = sanitize_name(raw_name)
    unique_name = f"id_{int(row.id):06d}__{base_name}"
    return base_name, unique_name


# ==========================================
# SMILES / Input DB
# ==========================================
def smiles_to_atoms(smiles: str) -> Atoms:
    """Converts a SMILES string to an ASE Atoms object using RDKit."""
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
    """
    Determines the input database.
    If SMILES are provided, creates a temporary DB and returns its path.
    Otherwise, returns the existing input_db path.
    """
    os.makedirs(workspace, exist_ok=True)

    if smiles_list:
        temp_db_path = os.path.join(workspace, 'temp_smiles_input.db')
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
        return os.path.join(workspace, 'all.db')

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
    """
    Merge order:
      key_value_pairs -> row.data -> atoms.info(fallback only)
    So row.data overrides key_value_pairs.
    """
    meta = {}

    kvp = _get_row_key_value_pairs(row)
    meta.update(kvp)

    data = _get_row_data(row)
    meta.update(data)

    try:
        if isinstance(getattr(atoms, "info", None), dict):
            for k, v in atoms.info.items():
                if k not in meta:
                    meta[k] = v
    except Exception:
        pass

    return meta


def _infer_charge(atoms: Atoms, meta: dict) -> int:
    """
    Charge priority:
      1) meta['charge'] if present
      2) infer from anion count keys: n_anion, n_anion_total, n_anions, n_anion_tot

    Inference rule:
      - if structure contains Li: charge = 1 - n_anion
      - else:                    charge = 0 - n_anion

    Fallback:
      - contains Li -> +1
      - else -> 0
    """
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

    out_db_path = os.path.join(workspace, 'optimized_all.db')
    fail_log_path = os.path.join(workspace, 'failed_jobs.log')
    shard_db_dir = os.path.join(workspace, 'worker_db_shards')
    out_xyz_dir = os.path.join(workspace, 'optimized_xyz_all') if write_xyz else None

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


def _build_worker_db_paths(shard_db_dir: str, n_workers: int) -> List[str]:
    return [
        os.path.join(shard_db_dir, f"worker_{i:03d}.db")
        for i in range(int(n_workers))
    ]


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


def _build_worker_gpu_plan(
    device: str,
    gpus: Optional[List[str]],
    workers_per_gpu: int,
    cpu_workers: Optional[int]
):
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
    """
    主进程先准备一次模型相关缓存，避免多进程首次并发初始化时互相抢。
    """
    from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies
    _ = get_isolated_atomic_energies(model_name, workspace)


# ==========================================
# Parent-death protection
# ==========================================
def _best_effort_set_pdeathsig():
    """
    Linux 下尽量设置 parent-death signal。
    失败也无所谓，后面还有 watchdog 线程兜底。
    """
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
    """
    守护线程：如果父进程消失或被 reparent 到 1，则强制退出。
    """
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
    """
    注意：
    1. 必须先设 CUDA_VISIBLE_DEVICES
    2. 再 import torch
    """
    _set_thread_env(cpu_threads)

    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import torch

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
                torch.cuda.set_device(0)
            except Exception:
                pass
    except Exception:
        pass


def _load_fairchem_calculator(checkpoint_path: str, model_name: str, workspace: str, device: str):
    """
    fairchem 这里 device 必须是 'cpu' 或 'cuda'
    """
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
def _task_feeder(
    input_db: str,
    task_queue,
    n_workers: int,
    result_queue,
):
    """
    动态投喂任务，避免一次性把所有 row id 存进内存。
    """
    try:
        with connect(input_db) as src_db:
            for row in src_db.select():
                task_queue.put(int(row.id))

        for _ in range(n_workers):
            task_queue.put(None)

        result_queue.put({"type": "feeder_done"})

    except Exception as e:
        result_queue.put({
            "type": "feeder_error",
            "error": f"Task feeder failed: {e}",
            "traceback": traceback.format_exc(),
        })


# ==========================================
# Merge shard DBs -> final DB
# ==========================================
def _merge_worker_dbs(
    worker_db_paths: List[str],
    out_db_path: str,
    write_xyz: bool = False,
    out_xyz_dir: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> int:
    """
    将每个 worker 写出的 shard DB 按 source_row_id 升序合并成最终 DB。
    串行合并，避免额外显存/并行开销。
    """
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
            shard_infos.append({
                "shard_idx": shard_idx,
                "path": path,
                "db": db,
                "iter": db.select(sort='source_row_id'),
            })

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
                        name = (
                            data.get("optimized_name", None)
                            or data.get("base_name", None)
                            or f"id_{src_id:06d}"
                        )
                        out_xyz = os.path.join(out_xyz_dir, f"{sanitize_name(name)}.xyz")
                        write(out_xyz, atoms)

                    merged += 1
                    pbar.update(1)

                    try:
                        next_row = next(it)
                        next_src_id = _extract_source_row_id_from_row(next_row)
                        heapq.heappush(heap, (next_src_id, shard_idx, next_row, it))
                    except StopIteration:
                        pass

                if verbose:
                    _safe_log(f"[MERGE] merged rows: {merged}", show_progress)
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
    worker_id: int,
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
):
    # 尽早启动“父死子亡”保护
    _best_effort_set_pdeathsig()
    _start_parent_watchdog(interval_sec=5.0)

    physical_device = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    fairchem_device = "cpu" if gpu_id is None else "cuda"
    input_db_abs = os.path.abspath(input_db)

    try:
        _configure_worker_runtime(cpu_threads_per_worker, gpu_id)

        calc = _load_fairchem_calculator(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            workspace=workspace,
            device=fairchem_device,
        )

        result_queue.put({
            "type": "worker_ready",
            "worker_id": worker_id,
            "device": physical_device,
            "worker_db": worker_db_path,
        })

    except Exception as e:
        result_queue.put({
            "type": "worker_fatal",
            "worker_id": worker_id,
            "device": physical_device,
            "error": f"Model init failed: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    try:
        with connect(input_db) as src_db, connect(worker_db_path) as out_db:
            while True:
                row_id = task_queue.get()
                if row_id is None:
                    task_queue.task_done()
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

                    atoms.info['charge'] = int(charge)
                    atoms.info['spin'] = int(spin)

                    base_name, safe_name = _derive_names(row, merged_meta)

                    result_queue.put({
                        "type": "started",
                        "worker_id": worker_id,
                        "device": physical_device,
                        "row_id": int(row.id),
                        "name": safe_name,
                        "charge": int(charge),
                        "spin": int(spin),
                    })

                    atoms.calc = calc

                    # 不写 trajectory，减少 IO
                    opt = LBFGS(atoms, logfile=None)
                    converged = bool(opt.run(fmax=fmax, steps=max_steps))

                    atoms.calc = None
                    nsteps = getattr(opt, "nsteps", None)

                    # 输出 data：保留原始元数据 + 溯源信息 + 优化信息
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
                        "_optimization": _jsonify({
                            "optimizer": "LBFGS",
                            "fmax": float(fmax),
                            "max_steps": int(max_steps),
                            "nsteps": _jsonify(nsteps),
                            "converged": bool(converged),
                            "device": physical_device,
                            "worker_id": int(worker_id),
                            "worker_db": os.path.basename(worker_db_path),
                        }),
                    }

                    # searchable top-level kvp：尽量保留原始可搜索标量
                    searchable_meta = dict(merged_meta)
                    searchable_meta.update({
                        "source_row_id": int(row.id),
                        "source_unique_id": str(getattr(row, "unique_id", "")),
                        "base_name": base_name,
                        "optimized_name": safe_name,
                        "source_db_name": os.path.basename(input_db_abs),
                        "opt_converged": bool(converged),
                        "worker_id": int(worker_id),
                        "worker_db": os.path.basename(worker_db_path),
                        "device_name": physical_device,
                    })
                    db_kvp = _prepare_db_key_value_pairs(searchable_meta)

                    # 每个 worker 写自己的 shard DB
                    out_db.write(atoms, data=record_data, **db_kvp)

                    result_queue.put({
                        "type": "done",
                        "worker_id": worker_id,
                        "device": physical_device,
                        "row_id": int(row.id),
                        "name": safe_name,
                        "charge": int(charge),
                        "spin": int(spin),
                        "converged": bool(converged),
                    })

                    del row, atoms, src_atoms_info, src_kvp, src_data, merged_meta
                    del record_data, searchable_meta, db_kvp

                except Exception as e:
                    result_queue.put({
                        "type": "error",
                        "worker_id": worker_id,
                        "device": physical_device,
                        "row_id": int(row_id),
                        "name": safe_name,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                finally:
                    task_queue.task_done()

    except Exception as e:
        result_queue.put({
            "type": "worker_fatal",
            "worker_id": worker_id,
            "device": physical_device,
            "error": f"Worker crashed: {e}",
            "traceback": traceback.format_exc(),
        })
    finally:
        try:
            import torch
            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        result_queue.put({
            "type": "worker_exit",
            "worker_id": worker_id,
            "device": physical_device,
        })


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
) -> str:
    """
    Main process:
      - aggregate progress
      - monitor worker status
      - merge per-worker shard DBs into final output DB

    Worker process:
      - bind to one physical GPU
      - load one FAIRChem model
      - pull tasks dynamically
      - optimize structures
      - write success results into its own shard DB
    """
    os.makedirs(workspace, exist_ok=True)

    # 1) Prepare workspace
    out_xyz_dir, shard_db_dir, out_db_path, fail_log_path = _prepare_workspace(
        workspace,
        write_xyz=write_xyz,
    )

    # 2) Prepare input source
    active_input_db = prepare_input_source(workspace, input_db, smiles)
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    # 3) Count jobs
    with connect(active_input_db) as src_db:
        total = src_db.count()

    if total == 0:
        with connect(out_db_path):
            pass
        if verbose:
            print(f"No rows found in input DB: {active_input_db}")
        return out_db_path

    # 4) Worker plan
    worker_gpu_plan = _build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    total_workers = len(worker_gpu_plan)

    if total_workers <= 0:
        raise RuntimeError("No workers planned.")

    worker_db_paths = _build_worker_db_paths(shard_db_dir, total_workers)

    # GPU 模式默认回滚到单 CPU 线程/worker；CPU 模式才自动分配
    if cpu_threads_per_worker is None:
        if str(device).lower().startswith("cuda"):
            cpu_threads_per_worker = 1
        else:
            ncpu = os.cpu_count() or 1
            cpu_threads_per_worker = max(1, ncpu // total_workers)

    _set_thread_env(cpu_threads_per_worker)

    if verbose:
        print("========== Parallel UMA Optimization ==========")
        print(f"Workspace              : {workspace}")
        print(f"Input DB               : {active_input_db}")
        print(f"Shard DB dir           : {shard_db_dir}")
        print(f"Final Output DB        : {out_db_path}")
        print(f"Write XYZ              : {write_xyz}")
        print(f"Keep worker DBs        : {keep_worker_dbs}")
        print(f"Checkpoint             : {checkpoint_path}")
        print(f"Model name             : {model_name}")
        print(f"Device                 : {device}")
        print(f"GPU plan               : {worker_gpu_plan}")
        print(f"Total workers          : {total_workers}")
        print(f"CPU threads / worker   : {cpu_threads_per_worker}")
        print(f"fmax                   : {fmax}")
        print(f"max_steps              : {max_steps}")
        print(f"Total molecules        : {total}")
        print("==============================================")

    # 5) Warmup model assets once in main
    if device.lower() != "cpu":
        if verbose:
            print("Preparing model assets...")
        _prepare_model_assets(model_name, workspace)

    # 6) Multiprocessing
    ctx = mp.get_context("spawn")

    if queue_size is None:
        task_qsize = max(64, total_workers * 4)
        result_qsize = max(32, total_workers * 4)
    else:
        task_qsize = int(queue_size)
        result_qsize = int(queue_size)

    task_queue = ctx.JoinableQueue(maxsize=task_qsize)
    result_queue = ctx.Queue(maxsize=result_qsize)

    workers = []
    for i, gpu_id in enumerate(worker_gpu_plan):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                i,
                gpu_id,
                active_input_db,
                worker_db_paths[i],
                checkpoint_path,
                model_name,
                workspace,
                fmax,
                max_steps,
                cpu_threads_per_worker,
                task_queue,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    # 7) Start feeder thread
    feeder_thread = threading.Thread(
        target=_task_feeder,
        args=(active_input_db, task_queue, len(workers), result_queue),
        daemon=True,
    )
    feeder_thread.start()

    # 8) Main loop
    completed = 0
    success = 0
    failed = 0
    running = {}
    normal_shutdown = False
    feeder_done = False

    with open(fail_log_path, "w", encoding="utf-8") as fail_fp:
        pbar = tqdm(
            total=total,
            desc="Optimizing",
            unit="mol",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        def _refresh_postfix():
            if not show_progress:
                return
            active_running = sum(running.values()) if running else 0
            postfix = {
                "ok": success,
                "fail": failed,
                "running": active_running,
            }
            gpu_items = [f"{k}:{running[k]}" for k in sorted(running.keys())]
            if gpu_items:
                postfix["gpu"] = ",".join(gpu_items)
            pbar.set_postfix(postfix)

        try:
            while completed < total:
                try:
                    msg = result_queue.get(timeout=1.0)
                except pyqueue.Empty:
                    alive = sum(int(p.is_alive()) for p in workers)
                    if alive == 0 and completed < total:
                        raise RuntimeError(
                            f"All workers exited early. Finished {completed}/{total}."
                        )
                    continue

                mtype = msg.get("type")

                if mtype == "feeder_done":
                    feeder_done = True
                    if verbose:
                        _safe_log("[FEED ] task feeder finished", show_progress)

                elif mtype == "feeder_error":
                    err_line = f"[FATAL][FEEDER] {msg['error']}"
                    fail_fp.write(err_line + "\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()

                    _safe_log(err_line, show_progress)
                    if verbose:
                        _safe_log(msg.get("traceback", ""), show_progress)

                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(err_line)

                elif mtype == "worker_ready":
                    if verbose:
                        _safe_log(
                            f"[READY][{msg['device']}][W{msg['worker_id']}] "
                            f"worker online | shard={msg.get('worker_db')}",
                            show_progress,
                        )

                elif mtype == "started":
                    running[msg["device"]] = running.get(msg["device"], 0) + 1
                    if verbose:
                        _safe_log(
                            f"[START][{msg['device']}][W{msg['worker_id']}] "
                            f"{msg['name']} | charge={msg['charge']} spin={msg['spin']}",
                            show_progress,
                        )
                    _refresh_postfix()

                elif mtype == "done":
                    completed += 1
                    success += 1
                    running[msg["device"]] = max(0, running.get(msg["device"], 0) - 1)

                    pbar.update(1)
                    _refresh_postfix()

                    if verbose:
                        _safe_log(
                            f"[DONE ][{msg['device']}][W{msg['worker_id']}] "
                            f"{msg['name']} | converged={msg['converged']}",
                            show_progress,
                        )

                elif mtype == "error":
                    completed += 1
                    failed += 1
                    running[msg["device"]] = max(0, running.get(msg["device"], 0) - 1)

                    err_name = msg.get("name") or f"id_{msg.get('row_id', -1)}"
                    err_line = (
                        f"[ERROR][{msg['device']}][W{msg['worker_id']}] "
                        f"{err_name} | row_id={msg.get('row_id')} | {msg['error']}"
                    )
                    fail_fp.write(err_line + "\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()

                    pbar.update(1)
                    _refresh_postfix()

                    _safe_log(err_line, show_progress)
                    if verbose:
                        _safe_log(msg.get("traceback", ""), show_progress)

                elif mtype == "worker_fatal":
                    err_line = (
                        f"[FATAL][{msg['device']}][W{msg['worker_id']}] {msg['error']}"
                    )
                    fail_fp.write(err_line + "\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()

                    _safe_log(err_line, show_progress)
                    if verbose:
                        _safe_log(msg.get("traceback", ""), show_progress)

                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(err_line)

                elif mtype == "worker_exit":
                    if verbose:
                        _safe_log(
                            f"[EXIT ][{msg['device']}][W{msg['worker_id']}] worker exited",
                            show_progress,
                        )

                else:
                    if verbose:
                        _safe_log(f"[WARN] Unknown message: {msg}", show_progress)

            normal_shutdown = True

        except KeyboardInterrupt:
            _safe_log("\n[Main] KeyboardInterrupt received, terminating workers...", show_progress)
            for p in workers:
                if p.is_alive():
                    p.terminate()
            raise
        finally:
            pbar.close()

    # 9) Shutdown / join
    try:
        if normal_shutdown:
            task_queue.join()
    except Exception:
        pass

    try:
        feeder_thread.join(timeout=5)
    except Exception:
        pass

    for p in workers:
        p.join(timeout=10)

    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    try:
        task_queue.close()
    except Exception:
        pass
    try:
        result_queue.close()
    except Exception:
        pass

    # 10) Merge shard DBs -> final DB
    merged_rows = 0
    if normal_shutdown:
        _safe_log("[Main] Merging worker shard DBs into final output DB...", show_progress)
        merged_rows = _merge_worker_dbs(
            worker_db_paths=worker_db_paths,
            out_db_path=out_db_path,
            write_xyz=write_xyz,
            out_xyz_dir=out_xyz_dir,
            verbose=verbose,
            show_progress=show_progress,
        )

        if merged_rows != success:
            _safe_log(
                f"[WARN] merged_rows({merged_rows}) != success({success}). Please inspect shard DBs.",
                show_progress,
            )

        if (not keep_worker_dbs) and os.path.exists(shard_db_dir):
            try:
                shutil.rmtree(shard_db_dir)
                if verbose:
                    _safe_log("[Main] Worker shard DBs removed.", show_progress)
            except Exception as e:
                _safe_log(f"[WARN] Failed to remove shard DB dir: {e}", show_progress)

    _safe_log(
        f"[Summary] total={total}, success={success}, failed={failed}, "
        f"merged={merged_rows}, feeder_done={feeder_done}, output_db={out_db_path}",
        show_progress,
    )
    if failed > 0:
        _safe_log(f"[Summary] failed log: {fail_log_path}", show_progress)

    return out_db_path


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="FAIRChem Parallel Optimization Script")

    parser.add_argument('--workspace', default=DEFAULT_WORKSPACE, help='Root output directory')

    # Input sources
    parser.add_argument('--input-db', default=None, help='Input ASE database')
    parser.add_argument('--smiles', nargs='*', default=None, help='Input SMILES string(s) to optimize')

    # Model
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Model name for atom refs cache')

    # Device / parallelism
    parser.add_argument('--device', default='cuda', help='Compute device: cuda / cuda:0 / cpu')
    parser.add_argument('--gpus', nargs='*', default=None, help='Physical GPU ids, e.g. --gpus 0 1')
    parser.add_argument('--workers-per-gpu', type=int, default=4,
                        help='Concurrent worker processes per GPU')
    parser.add_argument('--cpu-workers', type=int, default=None,
                        help='Only used when --device cpu')
    parser.add_argument('--cpu-threads-per-worker', type=int, default=None,
                        help='CPU threads used by each worker; default=1 on GPU mode')
    parser.add_argument('--queue-size', type=int, default=None,
                        help='Queue size for task/result queues (default: auto)')

    # Optimization
    parser.add_argument('--fmax', type=float, default=0.05, help='Force convergence criteria')
    parser.add_argument('--steps', type=int, default=200, dest='max_steps', help='Max optimization steps')

    # Output
    parser.add_argument('--write-xyz', action='store_true',
                        help='Also write optimized xyz files (default: False for max speed)')
    parser.add_argument('--keep-worker-dbs', action='store_true',
                        help='Keep per-worker shard DBs after final merge')

    # Logs
    parser.add_argument('--verbose', action='store_true', help='Show worker start/done logs')
    parser.add_argument('--no-progress', action='store_false', dest='progress',
                        help='Disable global progress bar')
    parser.set_defaults(progress=True)

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
    )

    print(f"\nOptimization finished. Output DB: {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()