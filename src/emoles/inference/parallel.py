import json
import os
import shutil

from emoles.inference.model_io import (
    default_fine_tune_ckpt_path,
    dptb_infer_to_lmdb_from_ase_db,
    merge_infer_lmdb_shards,
)
from emoles.utils.db import prepare_ase_db_worker_shards
from emoles.utils.parallel import (
    build_worker_gpu_plan,
    run_worker_pool,
    write_json_file,
)


def _dptb_lmdb_worker_main(worker_spec):
    log_path = os.path.join(worker_spec["worker_root"], "worker.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        def _log(message):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        worker_device = "cpu" if worker_spec.get("gpu_id") is None else "cuda"
        _log(
            f"[worker {worker_spec['worker_name']}] "
            f"gpu={worker_spec.get('gpu_id')} items={worker_spec['num_items']}"
        )

        dptb_infer_to_lmdb_from_ase_db(
            ase_db_path=worker_spec["shard_db_path"],
            out_path=worker_spec["out_path"],
            checkpoint_path=worker_spec["checkpoint_path"],
            max_items=worker_spec["num_items"],
            device=worker_device,
            infer_dir_name=worker_spec["infer_dir_name"],
            worker_name=worker_spec["worker_name"],
            has_overlap=worker_spec["has_overlap"],
            merge_shards=False,
            txn_batch_size=worker_spec["txn_batch_size"],
            cleanup_input_lmdb=worker_spec["cleanup_input_lmdb"],
        )
        _log(f"[worker {worker_spec['worker_name']}] finished")


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
):
    if limit is not None:
        max_items = limit

    ase_db_path = os.path.abspath(ase_db_path)
    out_path = os.path.abspath(out_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    pll_work_root = os.path.join(out_path, "pll_work")
    infer_root = os.path.join(out_path, infer_dir_name)
    if os.path.exists(infer_root):
        shutil.rmtree(infer_root)
    os.makedirs(infer_root, exist_ok=True)

    gpu_plan = build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    n_workers = len(gpu_plan)
    if n_workers <= 0:
        raise ValueError("No workers configured")

    worker_specs = prepare_ase_db_worker_shards(
        source_db_path=ase_db_path,
        work_root=pll_work_root,
        n_workers=n_workers,
        max_items=max_items,
    )
    gpu_plan = gpu_plan[: len(worker_specs)]

    for worker_spec, gpu_id in zip(worker_specs, gpu_plan):
        worker_spec["gpu_id"] = gpu_id
        worker_spec["out_path"] = out_path
        worker_spec["checkpoint_path"] = checkpoint_path
        worker_spec["infer_dir_name"] = infer_dir_name
        worker_spec["infer_lmdb_path"] = os.path.join(
            infer_root,
            f"{worker_spec['worker_name']}.lmdb",
        )
        worker_spec["has_overlap"] = bool(has_overlap)
        worker_spec["txn_batch_size"] = int(txn_batch_size)
        worker_spec["cleanup_input_lmdb"] = bool(cleanup_input_lmdb)

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
            "worker_specs": worker_specs,
        },
    )

    results = run_worker_pool(
        worker_specs=worker_specs,
        worker_main=_dptb_lmdb_worker_main,
        cpu_threads_per_worker=cpu_threads_per_worker,
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
