import json
import multiprocessing as mp
import os
import queue as pyqueue
import shutil
import time
import traceback

from tqdm import tqdm

from emoles.inference.model_io import (
    _build_gamma_projectors,
    _iter_predicted_batches,
    _prepare_dptb_model,
    _prepare_reference_loader,
    ase_db_2_dummy_dptb_lmdb,
    default_fine_tune_ckpt_path,
    merge_infer_lmdb_shards,
    save_info_2_lmdb,
)
from emoles.utils.db import (
    open_lmdb_environment,
    prepare_ase_db_worker_assignments,
    reset_lmdb_directory,
)
from emoles.utils.parallel import (
    build_worker_gpu_plan,
    configure_worker_env,
    query_gpu_mem_mb,
    set_thread_env,
    write_json_file,
)


def _safe_log(message, show_progress=True):
    if show_progress:
        tqdm.write(str(message))
    else:
        print(message)


def _count_lmdb_entries(lmdb_path):
    if not os.path.isdir(lmdb_path):
        return 0
    db_env = open_lmdb_environment(lmdb_path, readonly=True)
    try:
        with db_env.begin() as txn:
            return int(txn.stat()["entries"])
    finally:
        db_env.close()


def _count_committed_entries(worker_specs):
    return sum(_count_lmdb_entries(worker_spec["infer_lmdb_path"]) for worker_spec in worker_specs)


def _load_worker_items(items_path):
    with open(items_path, "r", encoding="utf-8") as f_obj:
        payload = json.load(f_obj)
    return payload.get("items", [])


def _strip_worker_spec(worker_spec):
    return {
        "worker_id": int(worker_spec["worker_id"]),
        "worker_name": worker_spec["worker_name"],
        "worker_root": worker_spec["worker_root"],
        "items_path": worker_spec["items_path"],
        "num_items": int(worker_spec["num_items"]),
        "source_idx_min": worker_spec.get("source_idx_min"),
        "source_idx_max": worker_spec.get("source_idx_max"),
        "gpu_id": worker_spec.get("gpu_id"),
        "infer_lmdb_path": worker_spec["infer_lmdb_path"],
        "input_lmdb_root": worker_spec["input_lmdb_root"],
    }


def _worker_manifest_payload(worker_spec, device_name, basis, r_max, entries, second_per_item):
    return {
        "ase_db_path": worker_spec["ase_db_path"],
        "checkpoint_path": worker_spec["checkpoint_path"],
        "device": device_name,
        "entries": int(entries),
        "worker_name": worker_spec["worker_name"],
        "shard_path": worker_spec["infer_lmdb_path"],
        "infer_root": worker_spec["infer_root"],
        "has_overlap": bool(worker_spec["has_overlap"]),
        "basis": basis,
        "r_max": r_max,
        "second_per_item": float(second_per_item),
        "items_path": worker_spec["items_path"],
        "cleanup_input_lmdb": bool(worker_spec["cleanup_input_lmdb"]),
        "key_field": "source_row_id",
    }


def _run_dptb_slot_worker(slot_id, attempt, worker_spec, cpu_threads_per_worker, result_queue):
    pid = os.getpid()
    gpu_id = worker_spec.get("gpu_id")
    logical_device = "cpu" if gpu_id is None else "cuda"
    device_name = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    log_path = os.path.join(worker_spec["worker_root"], "worker.log")

    with open(log_path, "a", encoding="utf-8") as log_file:
        def _log(message):
            line = str(message)
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

        try:
            configure_worker_env(
                gpu_id=gpu_id,
                cpu_threads_per_worker=cpu_threads_per_worker,
            )

            init_start = time.time()
            model, device, basis, r_max = _prepare_dptb_model(
                checkpoint_path=worker_spec["checkpoint_path"],
                device=logical_device,
            )
            projectors = _build_gamma_projectors(
                model=model,
                device=device,
                has_overlap=worker_spec["has_overlap"],
            )
            init_s = time.time() - init_start

            result_queue.put(
                {
                    "type": "worker_ready",
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                    "worker_name": worker_spec["worker_name"],
                    "init_s": float(init_s),
                    "num_items": int(worker_spec["num_items"]),
                }
            )
            _log(
                f"[READY][{device_name}][S{slot_id:03d}] "
                f"attempt={attempt} init_s={init_s:.1f} items={worker_spec['num_items']}"
            )

            items = _load_worker_items(worker_spec["items_path"])
            input_records = ase_db_2_dummy_dptb_lmdb(
                ase_db_path=worker_spec["ase_db_path"],
                dptb_lmdb_path=worker_spec["input_lmdb_root"],
                txn_batch_size=worker_spec["input_txn_batch_size"],
                items=items,
            )
            reference_loader = _prepare_reference_loader(
                lmdb_path=worker_spec["input_lmdb_root"],
                basis=basis,
                r_max=r_max,
            )

            reset_lmdb_directory(worker_spec["infer_lmdb_path"])
            output_env = open_lmdb_environment(worker_spec["infer_lmdb_path"])
            processed_items = 0
            infer_start = time.time()
            txn = output_env.begin(write=True)
            try:
                for idx, predicted_data in _iter_predicted_batches(
                    reference_loader=reference_loader,
                    model=model,
                    device=device,
                    max_items=None,
                ):
                    source_metadata = (
                        input_records[idx]
                        if idx < len(input_records)
                        else {"source_idx": idx, "source_row_id": None, "sample_id": idx}
                    )
                    save_info_2_lmdb(
                        txn=txn,
                        idx=idx,
                        source_metadata=source_metadata,
                        batch_info=predicted_data,
                        model=model,
                        device=device,
                        has_overlap=worker_spec["has_overlap"],
                        projectors=projectors,
                    )
                    processed_items += 1
                    if processed_items % max(1, int(worker_spec["txn_batch_size"])) == 0:
                        txn.commit()
                        txn = output_env.begin(write=True)
                txn.commit()
            except Exception:
                txn.abort()
                raise
            finally:
                output_env.close()

            if worker_spec["cleanup_input_lmdb"] and os.path.exists(worker_spec["input_lmdb_root"]):
                shutil.rmtree(worker_spec["input_lmdb_root"])

            second_per_item = (time.time() - infer_start) / max(1, processed_items)
            write_json_file(
                os.path.join(worker_spec["infer_lmdb_path"], "manifest.json"),
                _worker_manifest_payload(
                    worker_spec=worker_spec,
                    device_name=device_name,
                    basis=basis,
                    r_max=r_max,
                    entries=processed_items,
                    second_per_item=second_per_item,
                ),
            )
            result_queue.put(
                {
                    "type": "worker_done",
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                    "worker_name": worker_spec["worker_name"],
                    "entries": int(processed_items),
                    "second_per_item": float(second_per_item),
                }
            )
            _log(
                f"[DONE ][{device_name}][S{slot_id:03d}] "
                f"attempt={attempt} entries={processed_items} second_per_item={second_per_item:.4f}"
            )
        except Exception as exc:
            _log(f"[FATAL][{device_name}][S{slot_id:03d}] {exc!r}")
            _log(traceback.format_exc())
            result_queue.put(
                {
                    "type": "worker_fatal",
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                    "worker_name": worker_spec["worker_name"],
                    "error": repr(exc),
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


def dptb_infer_to_lmdb_from_ase_db_pll(
    ase_db_path,
    out_path,
    checkpoint_path=default_fine_tune_ckpt_path,
    max_items=None,
    device="cuda",
    gpus=None,
    workers_per_gpu=3,
    cpu_workers=None,
    cpu_threads_per_worker=None,
    infer_dir_name="infer",
    has_overlap=False,
    merge_shards=False,
    txn_batch_size=32,
    cleanup_input_lmdb=True,
    limit=None,
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
    progress_scan_interval_sec=5.0,
    show_progress=True,
    verbose=False,
):
    if limit is not None:
        max_items = limit

    ase_db_path = os.path.abspath(ase_db_path)
    out_path = os.path.abspath(out_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    pll_work_root = os.path.join(out_path, "pll_work")
    infer_root = os.path.join(out_path, infer_dir_name)
    input_root = os.path.join(out_path, "infer_input_lmdb")

    if os.path.exists(pll_work_root):
        shutil.rmtree(pll_work_root)
    if os.path.exists(infer_root):
        shutil.rmtree(infer_root)
    if os.path.exists(input_root):
        shutil.rmtree(input_root)
    os.makedirs(pll_work_root, exist_ok=True)
    os.makedirs(infer_root, exist_ok=True)
    os.makedirs(input_root, exist_ok=True)

    gpu_plan = build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    n_workers = len(gpu_plan)
    if n_workers <= 0:
        raise ValueError("No workers configured")

    worker_specs = prepare_ase_db_worker_assignments(
        source_db_path=ase_db_path,
        work_root=pll_work_root,
        n_workers=n_workers,
        max_items=max_items,
    )
    if not worker_specs:
        summary = {
            "infer_root": infer_root,
            "merged_path": None,
            "workers": 0,
            "worker_lmdb_paths": [],
            "results": [],
        }
        write_json_file(os.path.join(pll_work_root, "summary.json"), summary)
        write_json_file(
            os.path.join(infer_root, "manifest.json"),
            {
                "infer_root": infer_root,
                "merged_path": None,
                "worker_lmdb_paths": [],
                "workers": 0,
                "key_field": "source_row_id",
            },
        )
        return summary

    gpu_plan = gpu_plan[: len(worker_specs)]
    for worker_spec, gpu_id in zip(worker_specs, gpu_plan):
        worker_spec["gpu_id"] = gpu_id
        worker_spec["ase_db_path"] = ase_db_path
        worker_spec["checkpoint_path"] = checkpoint_path
        worker_spec["infer_root"] = infer_root
        worker_spec["infer_lmdb_path"] = os.path.join(
            infer_root,
            f"{worker_spec['worker_name']}.lmdb",
        )
        worker_spec["input_lmdb_root"] = os.path.join(
            input_root,
            worker_spec["worker_name"],
        )
        worker_spec["has_overlap"] = bool(has_overlap)
        worker_spec["txn_batch_size"] = int(txn_batch_size)
        worker_spec["input_txn_batch_size"] = max(32, int(txn_batch_size))
        worker_spec["cleanup_input_lmdb"] = bool(cleanup_input_lmdb)

    if cpu_threads_per_worker is None:
        ncpu = os.cpu_count() or 1
        cpu_threads_per_worker = max(1, ncpu // max(1, len(worker_specs)))
    set_thread_env(cpu_threads_per_worker)

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    write_json_file(
        os.path.join(pll_work_root, "run_manifest.json"),
        {
            "ase_db_path": ase_db_path,
            "out_path": out_path,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "gpus": [] if gpus is None else [int(gpu_id) for gpu_id in gpus],
            "workers_per_gpu": int(workers_per_gpu),
            "cpu_workers": cpu_workers,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "max_items": max_items,
            "warmup_workers_per_gpu": int(warmup_workers_per_gpu),
            "ramp_step_per_gpu": int(ramp_step_per_gpu),
            "ramp_interval_sec": float(ramp_interval_sec),
            "stable_window_sec": float(stable_window_sec),
            "gpu_headroom_gb": float(gpu_headroom_gb),
            "init_timeout_sec": float(init_timeout_sec),
            "startup_stagger_sec": float(startup_stagger_sec),
            "global_init_concurrency": int(global_init_concurrency),
            "init_concurrency_per_gpu": int(init_concurrency_per_gpu),
            "max_restarts_per_slot": int(max_restarts_per_slot),
            "worker_specs": [_strip_worker_spec(worker_spec) for worker_spec in worker_specs],
        },
    )

    total_items = sum(int(worker_spec["num_items"]) for worker_spec in worker_specs)
    total_slots = len(worker_specs)
    gpu_ids = sorted({gpu_id for gpu_id in gpu_plan if gpu_id is not None})

    slots_by_gpu = {}
    slot_rank_in_gpu = {}
    for slot_id, worker_spec in enumerate(worker_specs):
        gpu_id = worker_spec.get("gpu_id")
        slots_by_gpu.setdefault(gpu_id, []).append(slot_id)
    for gpu_id, slot_ids in slots_by_gpu.items():
        if gpu_id is None:
            continue
        for rank, slot_id in enumerate(slot_ids):
            slot_rank_in_gpu[slot_id] = rank

    warmup = int(max(0, min(warmup_workers_per_gpu, workers_per_gpu)))
    if str(device).lower() == "cpu":
        warmup = min(max(1, warmup_workers_per_gpu), total_slots)

    current_target_per_gpu = warmup if warmup > 0 else (1 if workers_per_gpu > 0 else 0)
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
    failures = []
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
        gpu_id = worker_specs[slot_id].get("gpu_id")
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
        gpu_id = worker_specs[slot_id].get("gpu_id")
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
            failure = {
                "worker_id": int(worker_specs[slot_id]["worker_id"]),
                "gpu_id": worker_specs[slot_id].get("gpu_id"),
                "status": "error",
                "error": f"slot exceeded max_restarts_per_slot after {why}",
                "exitcode": exitcode,
            }
            failures.append(failure)
            raise RuntimeError(json.dumps({"worker_failures": failures}, indent=2))
        if verbose:
            _safe_log(
                f"[RESPAWN][{_dev_name(worker_specs[slot_id].get('gpu_id'))}][S{slot_id:03d}] "
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

        proc = ctx.Process(
            target=_run_dptb_slot_worker,
            args=(
                slot_id,
                slot_attempt[slot_id],
                worker_specs[slot_id],
                cpu_threads_per_worker,
                result_queue,
            ),
        )
        proc.start()
        slot_proc[slot_id] = proc
        slot_pid[slot_id] = proc.pid
        slot_spawn_ts[slot_id] = time.time()
        slot_ready[slot_id] = False
        if verbose:
            _safe_log(
                f"[SPAWN][{_dev_name(worker_specs[slot_id].get('gpu_id'))}][S{slot_id:03d}] "
                f"attempt={slot_attempt[slot_id]} pid={proc.pid} reason={reason}",
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
                _safe_log(
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
            _safe_log(
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
        total=total_items,
        desc="DPTB PLL Infer",
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
                if now - last_progress_scan_ts >= float(progress_scan_interval_sec):
                    current_total = _count_committed_entries(worker_specs)
                    delta = current_total - committed_total
                    if delta:
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
                if verbose:
                    _safe_log(
                        f"[READY][{msg['device']}][S{slot_id:03d}] "
                        f"init_s={msg['init_s']:.1f} items={msg['num_items']}",
                        show_progress,
                    )
            elif msg_type == "worker_done":
                slot_id = int(msg["slot_id"])
                slot_result[slot_id] = {
                    "worker_id": int(worker_specs[slot_id]["worker_id"]),
                    "gpu_id": worker_specs[slot_id].get("gpu_id"),
                    "status": "ok",
                    "entries": int(msg["entries"]),
                    "second_per_item": float(msg["second_per_item"]),
                }
                completed_slots.add(slot_id)
                slot_ready[slot_id] = False
                if verbose:
                    _safe_log(
                        f"[DONE ][{msg['device']}][S{slot_id:03d}] "
                        f"entries={msg['entries']} second_per_item={msg['second_per_item']:.4f}",
                        show_progress,
                    )
            elif msg_type == "worker_fatal":
                slot_id = int(msg["slot_id"])
                _retire_process(slot_id)
                try:
                    _schedule_respawn(slot_id, why=msg.get("error", "worker_fatal"))
                except RuntimeError:
                    write_json_file(
                        os.path.join(pll_work_root, "worker_results.json"),
                        {"results": [result for result in slot_result if result is not None] + failures},
                    )
                    raise
    finally:
        current_total = _count_committed_entries(worker_specs)
        delta = current_total - committed_total
        if delta:
            progress_bar.update(delta)
            committed_total = current_total
        progress_bar.close()
        for proc in slot_proc:
            if proc is None:
                continue
            try:
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
            {
                "worker_id": int(worker_specs[slot_id]["worker_id"]),
                "gpu_id": worker_specs[slot_id].get("gpu_id"),
                "status": "error",
                "error": f"worker exited without completion result (exitcode={exitcode})",
            }
        )

    write_json_file(
        os.path.join(pll_work_root, "worker_results.json"),
        {"results": results},
    )

    failures = [result for result in results if result.get("status") != "ok"]
    if failures:
        raise RuntimeError(json.dumps({"worker_failures": failures}, indent=2))

    merged_path = None
    if merge_shards:
        merged_path = merge_infer_lmdb_shards(infer_root)

    summary = {
        "infer_root": infer_root,
        "merged_path": merged_path,
        "workers": len(worker_specs),
        "worker_lmdb_paths": [worker_spec["infer_lmdb_path"] for worker_spec in worker_specs],
        "results": results,
    }
    write_json_file(os.path.join(pll_work_root, "summary.json"), summary)
    write_json_file(
        os.path.join(infer_root, "manifest.json"),
        {
            "infer_root": infer_root,
            "merged_path": merged_path,
            "worker_lmdb_paths": [worker_spec["infer_lmdb_path"] for worker_spec in worker_specs],
            "workers": len(worker_specs),
            "key_field": "source_row_id",
        },
    )
    return summary
