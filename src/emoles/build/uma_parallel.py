import argparse
import multiprocessing as mp
import os
import queue as pyqueue
import shutil
import threading
import time
from typing import Dict, List, Optional, Tuple

from ase.db import connect
from tqdm import tqdm

from emoles.build.uma_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_WORKSPACE,
    prepare_input_source,
)
from emoles.build.uma_parallel_utils import (
    _build_worker_gpu_plan,
    _merge_worker_dbs,
    _prepare_model_assets,
    _prepare_workspace,
    _query_gpu_mem_mb,
    _safe_log,
    _set_thread_env,
    _task_feeder,
    _worker_loop,
)

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
