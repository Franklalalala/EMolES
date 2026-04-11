import json
import os
import shutil

from emoles.inference.model_io import (
    count_infer_worker_entries,
    default_fine_tune_ckpt_path,
    make_dptb_failure_result,
    merge_infer_lmdb_shards,
    populate_dptb_worker_specs,
    serialize_dptb_worker_spec,
    spawn_dptb_slot_process,
)
from emoles.utils.db import prepare_ase_db_worker_assignments
from emoles.utils.parallel import (
    build_worker_gpu_plan,
    run_ramped_slot_pool,
    set_thread_env,
    write_json_file,
)


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

    for path in (pll_work_root, infer_root, input_root):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    gpu_plan = build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    if not gpu_plan:
        raise ValueError("No workers configured")

    assignments = prepare_ase_db_worker_assignments(
        source_db_path=ase_db_path,
        work_root=pll_work_root,
        n_workers=len(gpu_plan),
        max_items=max_items,
    )
    if not assignments:
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

    worker_specs = populate_dptb_worker_specs(
        assignments,
        gpu_plan=gpu_plan,
        ase_db_path=ase_db_path,
        checkpoint_path=checkpoint_path,
        infer_root=infer_root,
        input_root=input_root,
        has_overlap=has_overlap,
        txn_batch_size=txn_batch_size,
        cleanup_input_lmdb=cleanup_input_lmdb,
    )

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
            "worker_specs": [serialize_dptb_worker_spec(worker_spec) for worker_spec in worker_specs],
        },
    )

    total_items = sum(int(worker_spec["num_items"]) for worker_spec in worker_specs)

    def _spawn_process(slot_id, attempt, result_queue, ctx):
        return spawn_dptb_slot_process(
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
        return make_dptb_failure_result(
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
        progress_scanner=lambda: count_infer_worker_entries(worker_specs),
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

    merged_path = merge_infer_lmdb_shards(infer_root) if merge_shards else None
    worker_lmdb_paths = [worker_spec["infer_lmdb_path"] for worker_spec in worker_specs]
    summary = {
        "infer_root": infer_root,
        "merged_path": merged_path,
        "workers": len(worker_specs),
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
            "workers": len(worker_specs),
            "key_field": "source_row_id",
        },
    )
    return summary


__all__ = [
    "dptb_infer_to_lmdb_from_ase_db_pll",
]
