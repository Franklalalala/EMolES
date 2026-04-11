import json
import os
import queue as pyqueue
import shutil
import time
import traceback

from ase.db import connect

from emoles.inference.model_io import (
    _build_gamma_projectors,
    _prepare_dptb_model,
    _prepare_reference_loader,
    ase_db_2_dummy_dptb_lmdb,
    default_fine_tune_ckpt_path,
    merge_infer_lmdb_shards,
    save_info_2_lmdb,
)
from emoles.utils.db import open_lmdb_environment, reset_lmdb_directory
from emoles.utils.parallel import (
    build_worker_gpu_plan,
    configure_worker_env,
    run_ase_db_task_queue_pool,
    set_thread_env,
    write_json_file,
)


def _count_lmdb_entries(lmdb_path):
    if not os.path.isdir(lmdb_path):
        return 0
    db_env = open_lmdb_environment(lmdb_path, readonly=True)
    try:
        with db_env.begin() as txn:
            return int(txn.stat()["entries"])
    finally:
        db_env.close()


def _read_worker_manifest(shard_path):
    manifest_path = os.path.join(shard_path, "manifest.json")
    if not os.path.exists(manifest_path):
        return {
            "shard_path": shard_path,
            "entries": _count_lmdb_entries(shard_path),
            "status": "ok",
        }
    with open(manifest_path, "r", encoding="utf-8") as f_obj:
        payload = json.load(f_obj)
    payload.setdefault("status", "ok")
    return payload


def _worker_manifest_payload(
    *,
    ase_db_path,
    checkpoint_path,
    device_name,
    entries,
    worker_name,
    shard_path,
    infer_root,
    has_overlap,
    basis,
    r_max,
    second_per_item,
    key_field="source_row_id",
):
    return {
        "ase_db_path": ase_db_path,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "device": device_name,
        "entries": int(entries),
        "worker_name": worker_name,
        "shard_path": shard_path,
        "infer_root": infer_root,
        "has_overlap": bool(has_overlap),
        "basis": basis,
        "r_max": r_max,
        "second_per_item": float(second_per_item),
        "key_field": key_field,
    }


def _spawn_dptb_worker(
    *,
    slot_id,
    attempt,
    gpu_id,
    ctx,
    task_queue,
    result_queue,
    stop_event,
    worker_root,
    ase_db_path,
    checkpoint_path,
    infer_root,
    has_overlap,
    cpu_threads_per_worker,
    txn_batch_size,
    cleanup_input_lmdb,
):
    worker_name = f"slot_{slot_id:03d}__a{attempt:04d}"
    shard_path = os.path.join(infer_root, f"{worker_name}.lmdb")
    input_lmdb_root = os.path.join(worker_root, f"{worker_name}_input")
    proc = ctx.Process(
        target=_run_dptb_queue_worker,
        args=(
            slot_id,
            attempt,
            gpu_id,
            {
                "ase_db_path": ase_db_path,
                "checkpoint_path": checkpoint_path,
                "infer_root": infer_root,
                "infer_lmdb_path": shard_path,
                "input_lmdb_root": input_lmdb_root,
                "worker_name": worker_name,
                "has_overlap": bool(has_overlap),
                "txn_batch_size": int(txn_batch_size),
                "cleanup_input_lmdb": bool(cleanup_input_lmdb),
            },
            cpu_threads_per_worker,
            task_queue,
            result_queue,
            stop_event,
        ),
    )
    proc.start()
    return proc, shard_path


def _run_dptb_queue_worker(
    slot_id,
    attempt,
    gpu_id,
    worker_spec,
    cpu_threads_per_worker,
    task_queue,
    result_queue,
    stop_event,
):
    pid = os.getpid()
    logical_device = "cpu" if gpu_id is None else "cuda"
    device_name = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    shard_path = worker_spec["infer_lmdb_path"]
    input_root = worker_spec["input_lmdb_root"]

    try:
        configure_worker_env(gpu_id=gpu_id, cpu_threads_per_worker=cpu_threads_per_worker)

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
                "slot_id": int(slot_id),
                "attempt": int(attempt),
                "pid": int(pid),
                "device": device_name,
                "worker_db": shard_path,
                "init_s": float(init_s),
                "mem_alloc_mb": mem_alloc_mb,
                "mem_reserved_mb": mem_reserved_mb,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "type": "worker_fatal",
                "stage": "init",
                "slot_id": int(slot_id),
                "attempt": int(attempt),
                "pid": int(pid),
                "device": device_name,
                "error": f"Model init failed: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    reset_lmdb_directory(shard_path)
    output_env = open_lmdb_environment(shard_path)
    txn = output_env.begin(write=True)
    processed_items = 0
    infer_start = time.time()

    try:
        with connect(worker_spec["ase_db_path"]) as src_db:
            while True:
                try:
                    row_id = task_queue.get(timeout=1.0)
                except pyqueue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if row_id is None:
                    break

                row_id = int(row_id)
                row = src_db.get(id=row_id)
                if row is None:
                    result_queue.put(
                        {
                            "type": "error",
                            "slot_id": int(slot_id),
                            "attempt": int(attempt),
                            "pid": int(pid),
                            "device": device_name,
                            "row_id": row_id,
                            "name": f"id_{row_id:06d}",
                            "error": f"ASE row id not found: {row_id}",
                            "traceback": "",
                        }
                    )
                    continue

                safe_name = f"id_{row_id:06d}"
                temp_input_root = os.path.join(input_root, safe_name)
                try:
                    input_records = ase_db_2_dummy_dptb_lmdb(
                        ase_db_path=worker_spec["ase_db_path"],
                        dptb_lmdb_path=temp_input_root,
                        txn_batch_size=1,
                        items=[
                            {
                                "source_idx": row_id,
                                "source_row_id": row_id,
                                "sample_id": row_id,
                            }
                        ],
                    )
                    reference_loader = _prepare_reference_loader(
                        lmdb_path=temp_input_root,
                        basis=basis,
                        r_max=r_max,
                    )

                    for idx, ref_batch in enumerate(reference_loader):
                        from dptb.data import AtomicData
                        import torch

                        source_metadata = input_records[idx]
                        result_queue.put(
                            {
                                "type": "started",
                                "slot_id": int(slot_id),
                                "attempt": int(attempt),
                                "pid": int(pid),
                                "device": device_name,
                                "row_id": int(source_metadata["source_row_id"]),
                                "name": safe_name,
                            }
                        )
                        batch = AtomicData.to_AtomicDataDict(ref_batch.to(device))
                        with torch.no_grad():
                            predicted_data = model(batch)
                        save_info_2_lmdb(
                            txn=txn,
                            idx=processed_items,
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
                        result_queue.put(
                            {
                                "type": "done",
                                "slot_id": int(slot_id),
                                "attempt": int(attempt),
                                "pid": int(pid),
                                "device": device_name,
                                "row_id": int(source_metadata["source_row_id"]),
                                "name": safe_name,
                            }
                        )
                except Exception as exc:
                    result_queue.put(
                        {
                            "type": "error",
                            "slot_id": int(slot_id),
                            "attempt": int(attempt),
                            "pid": int(pid),
                            "device": device_name,
                            "row_id": row_id,
                            "name": safe_name,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                finally:
                    if os.path.exists(temp_input_root):
                        shutil.rmtree(temp_input_root, ignore_errors=True)

        txn.commit()
        write_json_file(
            os.path.join(shard_path, "manifest.json"),
            _worker_manifest_payload(
                ase_db_path=worker_spec["ase_db_path"],
                checkpoint_path=worker_spec["checkpoint_path"],
                device_name=device_name,
                entries=processed_items,
                worker_name=worker_spec["worker_name"],
                shard_path=shard_path,
                infer_root=worker_spec["infer_root"],
                has_overlap=worker_spec["has_overlap"],
                basis=basis,
                r_max=r_max,
                second_per_item=(time.time() - infer_start) / max(1, processed_items),
            ),
        )
    except Exception as exc:
        txn.abort()
        result_queue.put(
            {
                "type": "worker_fatal",
                "stage": "run",
                "slot_id": int(slot_id),
                "attempt": int(attempt),
                "pid": int(pid),
                "device": device_name,
                "error": f"Worker crashed: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        output_env.close()
        if worker_spec["cleanup_input_lmdb"] and os.path.exists(input_root):
            shutil.rmtree(input_root, ignore_errors=True)
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
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                }
            )
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
    del progress_scan_interval_sec
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

    with connect(ase_db_path) as src_db:
        total_rows = src_db.count()
    if max_items is not None:
        total_rows = min(int(total_rows), int(max_items))
    if total_rows <= 0:
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

    worker_gpu_plan = build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    if not worker_gpu_plan:
        raise ValueError("No workers configured")

    if cpu_threads_per_worker is None:
        ncpu = os.cpu_count() or 1
        cpu_threads_per_worker = max(1, ncpu // max(1, len(worker_gpu_plan)))
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
        },
    )

    def _spawn_worker(slot_id, attempt, gpu_id, ctx, task_queue, result_queue, stop_event):
        return _spawn_dptb_worker(
            slot_id=slot_id,
            attempt=attempt,
            gpu_id=gpu_id,
            ctx=ctx,
            task_queue=task_queue,
            result_queue=result_queue,
            stop_event=stop_event,
            worker_root=input_root,
            ase_db_path=ase_db_path,
            checkpoint_path=checkpoint_path,
            infer_root=infer_root,
            has_overlap=has_overlap,
            cpu_threads_per_worker=cpu_threads_per_worker,
            txn_batch_size=txn_batch_size,
            cleanup_input_lmdb=cleanup_input_lmdb,
        )

    run_state = run_ase_db_task_queue_pool(
        input_db=ase_db_path,
        total_tasks=total_rows,
        worker_gpu_plan=worker_gpu_plan,
        spawn_worker=_spawn_worker,
        failure_log_path=os.path.join(pll_work_root, "failed_jobs.log"),
        progress_desc="DPTB PLL Infer",
        show_progress=show_progress,
        verbose=verbose,
        warmup_workers_per_gpu=warmup_workers_per_gpu,
        workers_per_gpu=workers_per_gpu,
        ramp_step_per_gpu=ramp_step_per_gpu,
        ramp_interval_sec=ramp_interval_sec,
        stable_window_sec=stable_window_sec,
        gpu_headroom_gb=gpu_headroom_gb,
        init_timeout_sec=init_timeout_sec,
        startup_stagger_sec=startup_stagger_sec,
        global_init_concurrency=global_init_concurrency,
        init_concurrency_per_gpu=init_concurrency_per_gpu,
        max_restarts_per_slot=max_restarts_per_slot,
        max_items=max_items,
    )

    worker_lmdb_paths = list(dict.fromkeys(run_state["spawn_artifacts"]))
    results = [_read_worker_manifest(path) for path in worker_lmdb_paths if os.path.exists(path)]
    write_json_file(os.path.join(pll_work_root, "worker_results.json"), {"results": results})

    failures = list(run_state["fail_ids"])
    if failures:
        raise RuntimeError(
            json.dumps(
                {
                    "failed_rows": failures,
                    "failed_log_path": os.path.join(pll_work_root, "failed_jobs.log"),
                },
                indent=2,
            )
        )

    merged_path = None
    if merge_shards:
        merged_path = merge_infer_lmdb_shards(infer_root)

    summary = {
        "infer_root": infer_root,
        "merged_path": merged_path,
        "workers": len(worker_gpu_plan),
        "worker_lmdb_paths": worker_lmdb_paths,
        "results": results,
    }
    write_json_file(os.path.join(pll_work_root, "summary.json"), summary)
    write_json_file(
        os.path.join(infer_root, "manifest.json"),
        {
            "infer_root": infer_root,
            "merged_path": merged_path,
            "worker_lmdb_paths": worker_lmdb_paths,
            "workers": len(worker_gpu_plan),
            "key_field": "source_row_id",
        },
    )
    return summary
