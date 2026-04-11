import json
import multiprocessing as mp
import os
import queue as pyqueue
import subprocess
import threading
import time
import traceback

from tqdm import tqdm


def resolve_gpu_ids(device="cuda", gpus=None):
    dev = str(device).lower().strip()
    if dev == "cpu":
        return []
    if gpus:
        return [int(x) for x in gpus]
    if dev.startswith("cuda:"):
        return [int(dev.split(":")[1])]
    try:
        import torch

        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0
    if n_gpus <= 0:
        raise RuntimeError("No CUDA GPU detected, but device is not cpu.")
    return list(range(n_gpus))


def build_worker_gpu_plan(
    device="cuda",
    gpus=None,
    workers_per_gpu=1,
    cpu_workers=None,
    n_tasks=None,
):
    dev = str(device).lower().strip()
    if dev == "cpu":
        plan = [None] * max(1, int(cpu_workers or n_tasks or 1))
    else:
        if int(workers_per_gpu) <= 0:
            raise ValueError("workers_per_gpu must be >= 1")
        plan = []
        for gpu_id in resolve_gpu_ids(device=device, gpus=gpus):
            plan.extend([int(gpu_id)] * int(workers_per_gpu))
    if n_tasks is not None:
        return plan[: int(n_tasks)]
    return plan


def set_thread_env(cpu_threads_per_worker=None):
    if cpu_threads_per_worker is None or int(cpu_threads_per_worker) <= 0:
        return
    value = str(int(cpu_threads_per_worker))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[key] = value


def configure_worker_env(gpu_id=None, cpu_threads_per_worker=None):
    set_thread_env(cpu_threads_per_worker)
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

    try:
        import torch

        if cpu_threads_per_worker is not None and int(cpu_threads_per_worker) > 0:
            try:
                torch.set_num_threads(int(cpu_threads_per_worker))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                precision = os.environ.get("EMOLES_FLOAT32_MATMUL_PRECISION", "highest")
                torch.set_float32_matmul_precision(precision)
                allow_tf32 = str(precision).lower() != "highest"
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = allow_tf32
            except Exception:
                pass
        if gpu_id is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
    except Exception:
        pass


def query_gpu_mem_mb(gpu_id):
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
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
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        parts = [part.strip() for part in output.split(",")]
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None
    return None


def write_json_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_obj:
        json.dump(payload, f_obj, indent=2, ensure_ascii=False)


def safe_log(message, show_progress=True):
    if show_progress:
        tqdm.write(str(message))
    else:
        print(message)


def feed_ase_db_task_queue(input_db, task_queue, result_queue, stop_event, max_items=None):
    try:
        from ase.db import connect

        with connect(input_db) as src_db:
            for idx, row in enumerate(src_db.select()):
                if max_items is not None and idx >= int(max_items):
                    break
                if stop_event.is_set():
                    break
                row_id = int(row.id)
                while True:
                    if stop_event.is_set():
                        break
                    try:
                        task_queue.put(row_id, timeout=1.0)
                        break
                    except pyqueue.Full:
                        continue
        result_queue.put({"type": "feeder_done"})
    except Exception as exc:
        result_queue.put(
            {
                "type": "feeder_error",
                "error": f"Task feeder failed: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _worker_entry(worker_main, worker_spec, cpu_threads_per_worker, result_queue):
    try:
        configure_worker_env(
            gpu_id=worker_spec.get("gpu_id"),
            cpu_threads_per_worker=cpu_threads_per_worker,
        )
        worker_main(worker_spec)
        result_queue.put(
            {
                "worker_id": int(worker_spec["worker_id"]),
                "gpu_id": worker_spec.get("gpu_id"),
                "status": "ok",
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": int(worker_spec["worker_id"]),
                "gpu_id": worker_spec.get("gpu_id"),
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_worker_pool(
    worker_specs,
    worker_main,
    cpu_threads_per_worker=None,
    poll_interval_sec=5.0,
):
    if not worker_specs:
        return []

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    recorded_worker_ids = set()

    for worker_spec in worker_specs:
        proc = ctx.Process(
            target=_worker_entry,
            args=(worker_main, worker_spec, cpu_threads_per_worker, result_queue),
        )
        proc.start()
        processes.append((worker_spec, proc))

    results = []
    while True:
        try:
            msg = result_queue.get(timeout=float(poll_interval_sec))
            results.append(msg)
            recorded_worker_ids.add(int(msg["worker_id"]))
        except Exception:
            pass

        all_exited = all((not proc.is_alive()) for _, proc in processes)
        if all_exited and len(recorded_worker_ids) >= len(processes):
            break
        if all_exited:
            break

    for worker_spec, proc in processes:
        proc.join()
        if proc.exitcode not in (0, None):
            already_recorded = any(
                result["worker_id"] == worker_spec["worker_id"] for result in results
            )
            if not already_recorded:
                results.append(
                    {
                        "worker_id": worker_spec["worker_id"],
                        "gpu_id": worker_spec.get("gpu_id"),
                        "status": "error",
                        "error": f"worker exited with code {proc.exitcode}",
                    }
                )

    return results


def run_ase_db_task_queue_pool(
    *,
    input_db,
    total_tasks,
    worker_gpu_plan,
    spawn_worker,
    failure_log_path,
    progress_desc="Optimizing",
    show_progress=True,
    verbose=False,
    queue_size=None,
    warmup_workers_per_gpu=1,
    workers_per_gpu=1,
    ramp_step_per_gpu=1,
    ramp_interval_sec=60.0,
    stable_window_sec=60.0,
    gpu_headroom_gb=13.0,
    init_timeout_sec=900.0,
    startup_stagger_sec=0.2,
    global_init_concurrency=1,
    init_concurrency_per_gpu=1,
    max_restarts_per_slot=200,
    restart_backoff_sec=2.0,
    task_timeout_sec=None,
    max_items=None,
):
    total_slots = len(worker_gpu_plan)
    if total_slots <= 0:
        raise RuntimeError("No workers planned.")

    gpu_ids = sorted(set([gpu_id for gpu_id in worker_gpu_plan if gpu_id is not None]))
    device = "cpu" if len(gpu_ids) == 0 else "cuda"

    slots_by_gpu = {}
    slot_rank_in_gpu = {}
    for slot_id, gpu_id in enumerate(worker_gpu_plan):
        slots_by_gpu.setdefault(gpu_id, []).append(slot_id)
    for gpu_id, slot_ids in slots_by_gpu.items():
        if gpu_id is None:
            continue
        for rank, slot_id in enumerate(slot_ids):
            slot_rank_in_gpu[slot_id] = rank

    warmup = int(max(0, min(warmup_workers_per_gpu, workers_per_gpu)))
    if str(device).lower() == "cpu":
        warmup = min(warmup_workers_per_gpu, total_slots) if warmup_workers_per_gpu > 0 else total_slots

    ctx = mp.get_context("spawn")
    task_qsize = max(64, total_slots * 4) if queue_size is None else int(queue_size)
    task_queue = ctx.Queue(maxsize=task_qsize)
    result_queue = ctx.Queue()
    stop_event = ctx.Event()

    feeder_thread = threading.Thread(
        target=feed_ase_db_task_queue,
        args=(input_db, task_queue, result_queue, stop_event, max_items),
        daemon=True,
    )
    feeder_thread.start()

    slot_proc = [None] * total_slots
    slot_pid = [None] * total_slots
    slot_attempt = [0] * total_slots
    slot_spawn_ts = [0.0] * total_slots
    slot_ready = [False] * total_slots
    slot_next_spawn_ts = [0.0] * total_slots
    slot_restarts = [0] * total_slots

    inflight = {}
    finished, ok_set, fail_set = set(), set(), set()
    running = {}
    spawn_artifacts = []

    current_target_per_gpu = warmup if warmup > 0 else (1 if workers_per_gpu > 0 else 0)
    current_target_per_gpu = int(min(current_target_per_gpu, workers_per_gpu))
    if str(device).lower() == "cpu":
        current_target_per_gpu = total_slots

    last_failure_ts = 0.0
    last_ramp_ts = 0.0
    pending_spawn = []
    global_next_spawn_ts = 0.0
    feeder_done = False
    normal_shutdown = False

    def _dev_name(gpu_id):
        return "cpu" if gpu_id is None else f"cuda:{gpu_id}"

    def _is_slot_active(slot_id):
        gpu_id = worker_gpu_plan[slot_id]
        if str(device).lower() == "cpu":
            return True
        if gpu_id is None:
            return False
        return slot_rank_in_gpu.get(slot_id, 10**9) < current_target_per_gpu

    def _active_slots():
        return [slot_id for slot_id in range(total_slots) if _is_slot_active(slot_id)]

    def _gpu_mem_ok_for_ramp():
        stats = {}
        if str(device).lower() == "cpu":
            return True, stats
        headroom_mb = int(float(gpu_headroom_gb) * 1024)
        for gpu_id in gpu_ids:
            mem = query_gpu_mem_mb(gpu_id)
            if mem is None:
                return False, stats
            used, total_mb = mem
            stats[gpu_id] = (used, total_mb)
            if used >= (total_mb - headroom_mb):
                return False, stats
        return True, stats

    def _retire_process(slot_id):
        proc = slot_proc[slot_id]
        if proc is None:
            return
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=0.2)
        except Exception:
            pass

    def _log_fail(fail_fp, line):
        fail_fp.write(line + "\n")
        fail_fp.flush()
        safe_log(line, show_progress)

    def _schedule_respawn(fail_fp, slot_id, why, exitcode=None):
        nonlocal last_failure_ts
        last_failure_ts = time.time()

        if slot_id in inflight:
            row_id, _started_at, dev = inflight.pop(slot_id)
            running[dev] = max(0, running.get(dev, 0) - 1)
            while True:
                if stop_event.is_set():
                    break
                try:
                    task_queue.put(int(row_id), timeout=1.0)
                    break
                except pyqueue.Full:
                    continue

        slot_ready[slot_id] = False
        slot_restarts[slot_id] += 1
        slot_attempt[slot_id] += 1

        restarts = slot_restarts[slot_id]
        backoff = float(restart_backoff_sec) * (2 ** min(restarts, 6))
        slot_next_spawn_ts[slot_id] = time.time() + min(backoff, 120.0)
        if slot_restarts[slot_id] > int(max_restarts_per_slot):
            raise RuntimeError(
                f"slot {slot_id} exceeded max_restarts_per_slot after {why} (exitcode={exitcode})"
            )

        _log_fail(
            fail_fp,
            f"[RESPAWN][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
            f"restarts={restarts} next_in={slot_next_spawn_ts[slot_id] - time.time():.1f}s "
            f"why={why} exit={exitcode}",
        )

    def _count_global_inits():
        count = 0
        for slot_id in _active_slots():
            proc = slot_proc[slot_id]
            if proc is not None and (not slot_ready[slot_id]) and proc.is_alive():
                count += 1
        return count

    def _count_gpu_inits(gpu_id):
        count = 0
        for slot_id in slots_by_gpu.get(gpu_id, []):
            if not _is_slot_active(slot_id):
                continue
            proc = slot_proc[slot_id]
            if proc is not None and (not slot_ready[slot_id]) and proc.is_alive():
                count += 1
        return count

    def _can_spawn(slot_id):
        if global_init_concurrency > 0 and _count_global_inits() >= int(global_init_concurrency):
            return False
        gpu_id = worker_gpu_plan[slot_id]
        if gpu_id is not None and init_concurrency_per_gpu > 0:
            if _count_gpu_inits(int(gpu_id)) >= int(init_concurrency_per_gpu):
                return False
        return True

    def _spawn_slot(slot_id, reason=""):
        if stop_event.is_set():
            return False
        if len(finished) >= int(total_tasks):
            return False
        if not _is_slot_active(slot_id):
            return False
        if slot_restarts[slot_id] >= int(max_restarts_per_slot):
            return False
        if time.time() < slot_next_spawn_ts[slot_id]:
            return False

        proc = slot_proc[slot_id]
        if proc is not None and proc.is_alive():
            return False

        try:
            proc, artifact = spawn_worker(
                slot_id=slot_id,
                attempt=slot_attempt[slot_id],
                gpu_id=worker_gpu_plan[slot_id],
                ctx=ctx,
                task_queue=task_queue,
                result_queue=result_queue,
                stop_event=stop_event,
            )
        except Exception as exc:
            slot_proc[slot_id] = None
            slot_pid[slot_id] = None
            slot_ready[slot_id] = False
            raise RuntimeError(f"spawn_failed slot={slot_id}: {exc}") from exc
        slot_proc[slot_id] = proc
        slot_pid[slot_id] = proc.pid
        slot_spawn_ts[slot_id] = time.time()
        slot_ready[slot_id] = False
        if artifact is not None:
            spawn_artifacts.append(artifact)
        if verbose:
            safe_log(
                f"[SPAWN][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
                f"attempt={slot_attempt[slot_id]} pid={proc.pid} reason={reason}",
                show_progress,
            )
        return True

    def _monitor_slots(fail_fp):
        now = time.time()
        for slot_id in _active_slots():
            proc = slot_proc[slot_id]

            if proc is not None and (not proc.is_alive()):
                exitcode = proc.exitcode
                try:
                    proc.join(timeout=0.0)
                except Exception:
                    pass
                slot_proc[slot_id] = None
                slot_pid[slot_id] = None
                _schedule_respawn(fail_fp, slot_id, why="proc_dead", exitcode=exitcode)
                continue

            if proc is not None and (not slot_ready[slot_id]) and init_timeout_sec:
                if (now - slot_spawn_ts[slot_id]) > float(init_timeout_sec):
                    _log_fail(
                        fail_fp,
                        f"[KILL ][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
                        f"init_timeout t={now - slot_spawn_ts[slot_id]:.1f}s",
                    )
                    _retire_process(slot_id)
                    slot_proc[slot_id] = None
                    slot_pid[slot_id] = None
                    _schedule_respawn(fail_fp, slot_id, why="init_timeout", exitcode=None)
                    continue

            if slot_proc[slot_id] is None and slot_id not in pending_spawn:
                pending_spawn.append(slot_id)

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

        for slot_id in _active_slots():
            proc = slot_proc[slot_id]
            if proc is None or (not proc.is_alive()) or (not slot_ready[slot_id]):
                return

        ok_mem, stats = _gpu_mem_ok_for_ramp()
        if not ok_mem:
            if verbose and stats:
                safe_log(
                    "[RAMP ] skip (mem headroom) | "
                    + " ".join([f"gpu{gpu_id}:{used}/{total_mb}MB" for gpu_id, (used, total_mb) in stats.items()]),
                    show_progress,
                )
            return

        current_target_per_gpu = min(int(workers_per_gpu), int(current_target_per_gpu) + int(ramp_step_per_gpu))
        last_ramp_ts = now

        stats_text = ""
        if stats:
            stats_text = " | " + " ".join([f"gpu{gpu_id}:{used}/{total_mb}MB" for gpu_id, (used, total_mb) in stats.items()])
        safe_log(
            f"[RAMP ] target_workers_per_gpu -> {current_target_per_gpu}{stats_text}",
            show_progress,
        )

        for gpu_id in gpu_ids:
            for slot_id in slots_by_gpu.get(gpu_id, []):
                if _is_slot_active(slot_id) and slot_proc[slot_id] is None and slot_id not in pending_spawn:
                    pending_spawn.append(slot_id)

    def _is_stale_msg(msg):
        slot_id = msg.get("slot_id", None)
        if slot_id is None or (not (0 <= int(slot_id) < total_slots)):
            return True
        slot_id = int(slot_id)
        if msg.get("pid", None) != slot_pid[slot_id]:
            return True
        if msg.get("attempt", None) != slot_attempt[slot_id]:
            return True
        return False

    for slot_id in _active_slots():
        pending_spawn.append(slot_id)

    if verbose:
        safe_log(
            f"[INIT ] gpus={gpu_ids} warmup={warmup} target={current_target_per_gpu}/{workers_per_gpu} "
            f"global_init_conc={global_init_concurrency} per_gpu_init_conc={init_concurrency_per_gpu}",
            show_progress,
        )

    with open(failure_log_path, "w", encoding="utf-8") as fail_fp:
        pbar = tqdm(
            total=int(total_tasks),
            desc=str(progress_desc),
            unit="mol",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        last_monitor = 0.0

        def _refresh_postfix():
            if not show_progress:
                return
            alive = sum(1 for slot_id in _active_slots() if slot_proc[slot_id] is not None and slot_proc[slot_id].is_alive())
            active_running = sum(running.values()) if running else 0
            postfix = {
                "ok": len(ok_set),
                "fail": len(fail_set),
                "run": active_running,
                "init": _count_global_inits(),
                "alive": alive,
                "tgt": current_target_per_gpu,
            }
            gpu_items = [f"{key.split(':')[-1]}:{running[key]}" for key in sorted(running.keys())]
            if gpu_items:
                postfix["g"] = ",".join(gpu_items)
            pbar.set_postfix(postfix)

        try:
            while len(finished) < int(total_tasks):
                now = time.time()

                if now - last_monitor > 1.0:
                    last_monitor = now
                    if task_timeout_sec is not None:
                        timeout = float(task_timeout_sec)
                        for slot_id, (row_id, started_at, dev) in list(inflight.items()):
                            if (now - started_at) > timeout:
                                _log_fail(
                                    fail_fp,
                                    f"[KILL ][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
                                    f"task_timeout row={row_id} t={now - started_at:.1f}s",
                                )
                                _retire_process(slot_id)
                                slot_proc[slot_id] = None
                                slot_pid[slot_id] = None
                                _schedule_respawn(fail_fp, slot_id, why=f"task_timeout row={row_id}", exitcode=None)

                    _monitor_slots(fail_fp)
                    _maybe_ramp_up()
                    _refresh_postfix()

                if pending_spawn and now >= global_next_spawn_ts:
                    for index, slot_id in enumerate(list(pending_spawn)):
                        if time.time() < slot_next_spawn_ts[slot_id]:
                            continue
                        if not _can_spawn(slot_id):
                            continue
                        pending_spawn.pop(index)
                        try:
                            spawned = _spawn_slot(slot_id, reason="queue")
                        except RuntimeError as exc:
                            _schedule_respawn(fail_fp, slot_id, why=str(exc), exitcode=None)
                            spawned = False
                        if spawned:
                            global_next_spawn_ts = time.time() + float(startup_stagger_sec)
                        break

                try:
                    msg = result_queue.get(timeout=0.1)
                except pyqueue.Empty:
                    continue

                msg_type = msg.get("type")
                if msg_type in ("worker_ready", "started", "done", "error", "worker_fatal", "worker_exit"):
                    if _is_stale_msg(msg):
                        continue

                if msg_type == "feeder_done":
                    feeder_done = True
                elif msg_type == "feeder_error":
                    stop_event.set()
                    raise RuntimeError(f"[FATAL][FEEDER] {msg['error']}")
                elif msg_type == "worker_ready":
                    slot_id = int(msg["slot_id"])
                    slot_ready[slot_id] = True
                    if verbose:
                        safe_log(
                            f"[READY][{msg['device']}][S{slot_id:03d}] init_s={msg.get('init_s'):.1f} "
                            f"VRAM_alloc={msg.get('mem_alloc_mb')}MB resv={msg.get('mem_reserved_mb')}MB",
                            show_progress,
                        )
                elif msg_type == "started":
                    slot_id = int(msg["slot_id"])
                    dev = msg["device"]
                    row_id = int(msg["row_id"])
                    running[dev] = running.get(dev, 0) + 1
                    inflight[slot_id] = (row_id, time.time(), dev)
                    _refresh_postfix()
                elif msg_type == "done":
                    slot_id = int(msg["slot_id"])
                    dev = msg["device"]
                    row_id = int(msg["row_id"])
                    inflight.pop(slot_id, None)
                    running[dev] = max(0, running.get(dev, 0) - 1)
                    if row_id not in finished:
                        finished.add(row_id)
                        ok_set.add(row_id)
                        pbar.update(1)
                    _refresh_postfix()
                elif msg_type == "error":
                    slot_id = int(msg["slot_id"])
                    dev = msg["device"]
                    row_id = int(msg.get("row_id", -1))
                    inflight.pop(slot_id, None)
                    running[dev] = max(0, running.get(dev, 0) - 1)
                    fail_fp.write(f"[ERROR][{dev}][S{slot_id:03d}] row={row_id} | {msg['error']}\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()
                    if row_id not in finished:
                        finished.add(row_id)
                        fail_set.add(row_id)
                        pbar.update(1)
                    _refresh_postfix()
                elif msg_type == "worker_fatal":
                    slot_id = int(msg.get("slot_id", -1))
                    stage = msg.get("stage", "unknown")
                    dev_name = msg.get("device", "unknown")
                    fail_fp.write(f"[FATAL][{dev_name}][S{slot_id:03d}] stage={stage} | {msg['error']}\n")
                    fail_fp.write(msg.get("traceback", "") + "\n")
                    fail_fp.flush()
                    if 0 <= slot_id < total_slots and _is_slot_active(slot_id):
                        _retire_process(slot_id)
                        slot_proc[slot_id] = None
                        slot_pid[slot_id] = None
                        _schedule_respawn(fail_fp, slot_id, why=f"worker_fatal stage={stage}", exitcode=None)
                elif msg_type == "worker_exit":
                    pass

            normal_shutdown = True
        except KeyboardInterrupt:
            safe_log("\n[Main] KeyboardInterrupt received, terminating workers...", show_progress)
            stop_event.set()
            raise
        finally:
            pbar.close()

    stop_event.set()

    end_drain = time.time() + 1.0
    while time.time() < end_drain:
        try:
            result_queue.get(timeout=0.1)
        except pyqueue.Empty:
            pass

    for _ in range(total_slots * 2):
        try:
            task_queue.put_nowait(None)
        except Exception:
            break

    try:
        feeder_thread.join(timeout=3)
    except Exception:
        pass

    for slot_id in range(total_slots):
        proc = slot_proc[slot_id]
        if proc is None:
            continue
        try:
            proc.join(timeout=3)
        except Exception:
            pass

    for slot_id in range(total_slots):
        proc = slot_proc[slot_id]
        if proc is not None and proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=1)
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

    return {
        "ok_ids": sorted(ok_set),
        "fail_ids": sorted(fail_set),
        "feeder_done": bool(feeder_done),
        "normal_shutdown": bool(normal_shutdown),
        "spawn_artifacts": spawn_artifacts,
    }


def run_ramped_slot_pool(
    *,
    worker_specs,
    spawn_process,
    make_done_result,
    make_failure_result=None,
    device="cuda",
    workers_per_gpu=1,
    warmup_workers_per_gpu=1,
    ramp_step_per_gpu=1,
    ramp_interval_sec=60.0,
    stable_window_sec=60.0,
    gpu_headroom_gb=13.0,
    init_timeout_sec=900.0,
    startup_stagger_sec=0.2,
    global_init_concurrency=1,
    init_concurrency_per_gpu=1,
    max_restarts_per_slot=8,
    progress_total=0,
    progress_desc="PLL",
    progress_scan_interval_sec=5.0,
    progress_scanner=None,
    show_progress=True,
    verbose=False,
):
    if not worker_specs:
        return []

    if make_failure_result is None:
        def make_failure_result(slot_id, error, exitcode=None):
            worker_spec = worker_specs[slot_id]
            payload = {
                "worker_id": int(worker_spec["worker_id"]),
                "gpu_id": worker_spec.get("gpu_id"),
                "status": "error",
                "error": str(error),
            }
            if exitcode is not None:
                payload["exitcode"] = exitcode
            return payload

    worker_gpu_plan = [worker_spec.get("gpu_id") for worker_spec in worker_specs]
    total_slots = len(worker_specs)
    gpu_ids = sorted({gpu_id for gpu_id in worker_gpu_plan if gpu_id is not None})
    if str(device).lower() == "cpu":
        gpu_ids = []

    slots_by_gpu = {}
    slot_rank_in_gpu = {}
    for slot_id, gpu_id in enumerate(worker_gpu_plan):
        slots_by_gpu.setdefault(gpu_id, []).append(slot_id)
    for gpu_id, slot_ids in slots_by_gpu.items():
        if gpu_id is None:
            continue
        for rank, slot_id in enumerate(slot_ids):
            slot_rank_in_gpu[slot_id] = rank

    warmup = int(max(0, min(warmup_workers_per_gpu, workers_per_gpu)))
    if str(device).lower() == "cpu":
        warmup = min(max(1, int(warmup_workers_per_gpu)), total_slots)

    current_target_per_gpu = warmup if warmup > 0 else (1 if int(workers_per_gpu) > 0 else 0)
    current_target_per_gpu = min(int(workers_per_gpu), int(current_target_per_gpu))
    if str(device).lower() == "cpu":
        current_target_per_gpu = total_slots

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    slot_proc = [None] * total_slots
    slot_pid = [None] * total_slots
    slot_ready = [False] * total_slots
    slot_attempt = [0] * total_slots
    slot_spawn_ts = [0.0] * total_slots
    slot_next_spawn_ts = [0.0] * total_slots
    slot_restarts = [0] * total_slots
    slot_result = [None] * total_slots
    completed_slots = set()
    pending_spawn = []
    last_failure_ts = 0.0
    last_ramp_ts = 0.0
    global_next_spawn_ts = 0.0
    last_progress_scan_ts = 0.0
    committed_total = 0
    last_monitor_ts = 0.0

    def _dev_name(gpu_id):
        return "cpu" if gpu_id is None else f"cuda:{gpu_id}"

    def _is_slot_active(slot_id):
        if str(device).lower() == "cpu":
            return True
        gpu_id = worker_gpu_plan[slot_id]
        if gpu_id is None:
            return False
        return slot_rank_in_gpu.get(slot_id, 10**9) < current_target_per_gpu

    def _active_slots():
        return [slot_id for slot_id in range(total_slots) if _is_slot_active(slot_id)]

    def _active_slots_ready_for_ramp():
        for slot_id in _active_slots():
            if slot_id in completed_slots:
                continue
            proc = slot_proc[slot_id]
            if proc is None or (not proc.is_alive()) or (not slot_ready[slot_id]):
                return False
        return True

    def _count_global_inits():
        count = 0
        for slot_id in _active_slots():
            if slot_id in completed_slots:
                continue
            proc = slot_proc[slot_id]
            if proc is not None and proc.is_alive() and (not slot_ready[slot_id]):
                count += 1
        return count

    def _count_gpu_inits(gpu_id):
        count = 0
        for slot_id in slots_by_gpu.get(gpu_id, []):
            if not _is_slot_active(slot_id) or slot_id in completed_slots:
                continue
            proc = slot_proc[slot_id]
            if proc is not None and proc.is_alive() and (not slot_ready[slot_id]):
                count += 1
        return count

    def _gpu_mem_ok_for_ramp():
        if str(device).lower() == "cpu":
            return True, {}
        stats = {}
        headroom_mb = int(float(gpu_headroom_gb) * 1024)
        for gpu_id in gpu_ids:
            mem = query_gpu_mem_mb(gpu_id)
            if mem is None:
                return False, stats
            used, total_mb = mem
            stats[gpu_id] = (used, total_mb)
            if used >= (total_mb - headroom_mb):
                return False, stats
        return True, stats

    def _can_spawn(slot_id):
        if global_init_concurrency > 0 and _count_global_inits() >= int(global_init_concurrency):
            return False
        gpu_id = worker_gpu_plan[slot_id]
        if gpu_id is not None and init_concurrency_per_gpu > 0:
            if _count_gpu_inits(gpu_id) >= int(init_concurrency_per_gpu):
                return False
        return True

    def _retire_process(slot_id):
        proc = slot_proc[slot_id]
        if proc is None:
            return
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=0.2)
        except Exception:
            pass

    def _schedule_respawn(slot_id, why, exitcode=None):
        nonlocal last_failure_ts
        slot_ready[slot_id] = False
        slot_proc[slot_id] = None
        slot_pid[slot_id] = None
        slot_result[slot_id] = None
        slot_restarts[slot_id] += 1
        slot_attempt[slot_id] += 1
        last_failure_ts = time.time()
        delay = min(120.0, float(2.0 ** min(slot_restarts[slot_id], 6)))
        slot_next_spawn_ts[slot_id] = time.time() + delay
        if slot_restarts[slot_id] > int(max_restarts_per_slot):
            slot_result[slot_id] = make_failure_result(
                slot_id,
                f"slot exceeded max_restarts_per_slot after {why}",
                exitcode=exitcode,
            )
            completed_slots.add(slot_id)
            return
        if verbose:
            safe_log(
                f"[RESPAWN][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
                f"attempt={slot_attempt[slot_id]} why={why} exit={exitcode}",
                show_progress,
            )

    def _spawn_slot(slot_id, reason=""):
        if slot_id in completed_slots:
            return False
        if time.time() < slot_next_spawn_ts[slot_id]:
            return False
        proc = slot_proc[slot_id]
        if proc is not None and proc.is_alive():
            return False
        try:
            proc = spawn_process(
                slot_id=slot_id,
                attempt=slot_attempt[slot_id],
                result_queue=result_queue,
                ctx=ctx,
            )
        except Exception as exc:
            _schedule_respawn(slot_id, why=f"spawn_failed: {exc}", exitcode=None)
            return False
        slot_proc[slot_id] = proc
        slot_pid[slot_id] = getattr(proc, "pid", None)
        slot_spawn_ts[slot_id] = time.time()
        slot_ready[slot_id] = False
        if verbose:
            safe_log(
                f"[SPAWN][{_dev_name(worker_gpu_plan[slot_id])}][S{slot_id:03d}] "
                f"attempt={slot_attempt[slot_id]} pid={slot_pid[slot_id]} reason={reason}",
                show_progress,
            )
        return True

    def _is_stale_msg(msg):
        slot_id = msg.get("slot_id")
        if slot_id is None or int(slot_id) >= total_slots:
            return True
        slot_id = int(slot_id)
        if msg.get("pid") != slot_pid[slot_id]:
            return True
        if msg.get("attempt") != slot_attempt[slot_id]:
            return True
        return False

    def _monitor_slots():
        now = time.time()
        for slot_id in _active_slots():
            proc = slot_proc[slot_id]
            if proc is not None and (not proc.is_alive()):
                exitcode = proc.exitcode
                try:
                    proc.join(timeout=0.0)
                except Exception:
                    pass
                if slot_id in completed_slots and exitcode == 0:
                    slot_proc[slot_id] = None
                    slot_pid[slot_id] = None
                    continue
                _schedule_respawn(slot_id, why="proc_dead", exitcode=exitcode)
                continue

            if (
                proc is not None
                and proc.is_alive()
                and (not slot_ready[slot_id])
                and init_timeout_sec
                and (now - slot_spawn_ts[slot_id]) > float(init_timeout_sec)
            ):
                _retire_process(slot_id)
                _schedule_respawn(slot_id, why="init_timeout")
                continue

            if (
                slot_id not in completed_slots
                and slot_proc[slot_id] is None
                and slot_id not in pending_spawn
            ):
                pending_spawn.append(slot_id)

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
        if not _active_slots_ready_for_ramp():
            return
        ok_mem, stats = _gpu_mem_ok_for_ramp()
        if not ok_mem:
            if verbose and stats:
                safe_log(
                    "[RAMP ] skip (mem headroom) | "
                    + " ".join(
                        f"gpu{gpu_id}:{used}/{total_mb}MB"
                        for gpu_id, (used, total_mb) in stats.items()
                    ),
                    show_progress,
                )
            return
        current_target_per_gpu = min(
            int(workers_per_gpu),
            int(current_target_per_gpu) + int(ramp_step_per_gpu),
        )
        last_ramp_ts = now
        if verbose:
            safe_log(
                f"[RAMP ] target_workers_per_gpu -> {current_target_per_gpu}",
                show_progress,
            )
        for gpu_id in gpu_ids:
            for slot_id in slots_by_gpu.get(gpu_id, []):
                if (
                    _is_slot_active(slot_id)
                    and slot_id not in completed_slots
                    and slot_proc[slot_id] is None
                    and slot_id not in pending_spawn
                ):
                    pending_spawn.append(slot_id)

    for slot_id in _active_slots():
        pending_spawn.append(slot_id)

    progress_bar = tqdm(
        total=int(progress_total),
        desc=str(progress_desc),
        unit="mol",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    try:
        while len(completed_slots) < total_slots:
            now = time.time()
            if now - last_monitor_ts > 1.0:
                last_monitor_ts = now
                _monitor_slots()
                _maybe_ramp_up()
                if progress_scanner is not None and (
                    now - last_progress_scan_ts >= float(progress_scan_interval_sec)
                ):
                    current_total = int(progress_scanner())
                    delta = current_total - committed_total
                    if delta > 0:
                        progress_bar.update(delta)
                        committed_total = current_total
                    last_progress_scan_ts = now
                if show_progress:
                    progress_bar.set_postfix(
                        {
                            "ok": len(completed_slots),
                            "init": _count_global_inits(),
                            "alive": sum(
                                1
                                for slot_id in _active_slots()
                                if slot_proc[slot_id] is not None and slot_proc[slot_id].is_alive()
                            ),
                            "tgt": current_target_per_gpu,
                        }
                    )

            if pending_spawn and now >= global_next_spawn_ts:
                for index, slot_id in enumerate(list(pending_spawn)):
                    if time.time() < slot_next_spawn_ts[slot_id]:
                        continue
                    if not _can_spawn(slot_id):
                        continue
                    pending_spawn.pop(index)
                    if _spawn_slot(slot_id, reason="queue"):
                        global_next_spawn_ts = time.time() + float(startup_stagger_sec)
                    break

            try:
                msg = result_queue.get(timeout=0.1)
            except pyqueue.Empty:
                continue

            msg_type = msg.get("type")
            if msg_type in ("worker_ready", "worker_done", "worker_fatal") and _is_stale_msg(msg):
                continue

            if msg_type == "worker_ready":
                slot_id = int(msg["slot_id"])
                slot_ready[slot_id] = True
            elif msg_type == "worker_done":
                slot_id = int(msg["slot_id"])
                slot_result[slot_id] = make_done_result(slot_id, msg)
                completed_slots.add(slot_id)
                slot_ready[slot_id] = False
            elif msg_type == "worker_fatal":
                slot_id = int(msg["slot_id"])
                _retire_process(slot_id)
                _schedule_respawn(slot_id, why=msg.get("error", "worker_fatal"))
    finally:
        if progress_scanner is not None:
            current_total = int(progress_scanner())
            delta = current_total - committed_total
            if delta > 0:
                progress_bar.update(delta)
        progress_bar.close()
        for proc in slot_proc:
            if proc is None:
                continue
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.2)
            except Exception:
                pass

    results = []
    for slot_id, result in enumerate(slot_result):
        if result is not None:
            results.append(result)
            continue
        proc = slot_proc[slot_id]
        exitcode = None if proc is None else proc.exitcode
        results.append(
            make_failure_result(
                slot_id,
                f"worker exited without completion result (exitcode={exitcode})",
                exitcode=exitcode,
            )
        )
    return results
