import json
import os
import shutil
import time
import traceback

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
    run_ramped_slot_pool,
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


def _slot_failure_result(worker_spec, error, exitcode=None):
    payload = {
        "worker_id": int(worker_spec["worker_id"]),
        "gpu_id": worker_spec.get("gpu_id"),
        "status": "error",
        "error": str(error),
    }
    if exitcode is not None:
        payload["exitcode"] = exitcode
    return payload


def _spawn_dptb_slot_process(
    *,
    slot_id,
    attempt,
    result_queue,
    ctx,
    worker_specs,
    cpu_threads_per_worker,
):
    proc = ctx.Process(
        target=_run_dptb_slot_worker,
        args=(
            slot_id,
            attempt,
            worker_specs[slot_id],
            cpu_threads_per_worker,
            result_queue,
        ),
    )
    proc.start()
    return proc


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

    def _spawn_process(slot_id, attempt, result_queue, ctx):
        return _spawn_dptb_slot_process(
            slot_id=slot_id,
            attempt=attempt,
            result_queue=result_queue,
            ctx=ctx,
            worker_specs=worker_specs,
            cpu_threads_per_worker=cpu_threads_per_worker,
        )

    def _make_done_result(slot_id, msg):
        return {
            "worker_id": int(worker_specs[slot_id]["worker_id"]),
            "gpu_id": worker_specs[slot_id].get("gpu_id"),
            "status": "ok",
            "entries": int(msg["entries"]),
            "second_per_item": float(msg["second_per_item"]),
        }

    def _make_failure_result(slot_id, error, exitcode=None):
        return _slot_failure_result(
            worker_specs[slot_id],
            error,
            exitcode=exitcode,
        )

    results = run_ramped_slot_pool(
        worker_specs=worker_specs,
        spawn_process=_spawn_process,
        make_done_result=_make_done_result,
        make_failure_result=_make_failure_result,
        device=device,
        workers_per_gpu=workers_per_gpu,
        warmup_workers_per_gpu=warmup_workers_per_gpu,
        ramp_step_per_gpu=ramp_step_per_gpu,
        ramp_interval_sec=ramp_interval_sec,
        stable_window_sec=stable_window_sec,
        gpu_headroom_gb=gpu_headroom_gb,
        init_timeout_sec=init_timeout_sec,
        startup_stagger_sec=startup_stagger_sec,
        global_init_concurrency=global_init_concurrency,
        init_concurrency_per_gpu=init_concurrency_per_gpu,
        max_restarts_per_slot=max_restarts_per_slot,
        progress_total=total_items,
        progress_desc="DPTB PLL Infer",
        progress_scan_interval_sec=progress_scan_interval_sec,
        progress_scanner=lambda: _count_committed_entries(worker_specs),
        show_progress=show_progress,
        verbose=verbose,
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
