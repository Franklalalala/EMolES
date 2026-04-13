import argparse
import json
import os
import shutil

from ase.db import connect
from dpdispatcher import Task

from emoles.inference.infer_entry import dm_infer_entry_from_lmdb
from emoles.inference.postprocess import (
    DEFAULT_SUMMARY_JSON_NAME,
    DEFAULT_UPDATED_ASE_DB_NAME,
    save_inference_summary_npz,
    write_dm_inference_ase_db,
    write_inference_summary_json,
)
from emoles.hpc.utils import (
    build_submission,
    launch_local_job_queue,
    prepare_cooking_dir,
    wait_for_local_jobs,
)
from emoles.utils import (
    configure_worker_env,
    extract_lmdb_records,
    prepare_ase_db_worker_shards,
    write_json_file,
)


def _collect_source_row_ids(shard_db_path):
    source_row_ids = []
    with connect(shard_db_path) as shard_db:
        for row in shard_db.select():
            data = getattr(row, "data", None) or {}
            source_row_id = data.get("source_row_id", row.id)
            source_row_ids.append(int(source_row_id))
    return source_row_ids


def _default_results_dir(infer_lmdb_path):
    infer_lmdb_path = os.path.abspath(infer_lmdb_path)
    if infer_lmdb_path.endswith(".lmdb"):
        infer_root = os.path.dirname(infer_lmdb_path)
    else:
        infer_root = infer_lmdb_path
    if os.path.basename(infer_root) == "infer":
        return os.path.join(os.path.dirname(infer_root), "results")
    return os.path.join(infer_root, "results")


def _count_ase_db_rows(ase_db_path):
    with connect(ase_db_path) as db:
        return db.count()


def _prepare_dm_infer_shards(
    main_db_path,
    infer_lmdb_path,
    cooking_path,
    n_workers,
    handler_inputs=None,
):
    handler_inputs = dict(handler_inputs or {})
    max_items = handler_inputs.get("max_items", None)
    n_save_cube_items = int(handler_inputs.get("n_save_cube_items", 5))

    worker_specs = prepare_ase_db_worker_shards(
        source_db_path=main_db_path,
        work_root=cooking_path,
        n_workers=n_workers,
        global_n_save_cube_items=n_save_cube_items,
        max_items=max_items,
    )

    for worker_spec in worker_specs:
        worker_results_path = os.path.join(worker_spec["worker_root"], "results")
        worker_infer_path = os.path.join(worker_spec["worker_root"], "infer.lmdb")
        os.makedirs(worker_results_path, exist_ok=True)

        source_row_ids = _collect_source_row_ids(worker_spec["shard_db_path"])
        extract_info = extract_lmdb_records(
            infer_lmdb_path,
            worker_infer_path,
            source_row_ids,
        )
        if extract_info["missing"]:
            raise KeyError(
                f"Missing LMDB records for worker {worker_spec['worker_name']}: {extract_info['missing'][:10]}"
            )

        worker_payload = dict(handler_inputs)
        worker_payload.update(
            {
                "abs_ase_path": os.path.abspath(worker_spec["shard_db_path"]),
                "infer_lmdb_path": os.path.abspath(worker_infer_path),
                "results_folder_path": os.path.abspath(worker_results_path),
                "max_items": int(worker_spec["num_items"]),
                "n_save_cube_items": int(worker_spec["n_save_cube_items"]),
            }
        )
        write_json_file(
            os.path.join(worker_spec["worker_root"], "handler_inputs.json"),
            worker_payload,
        )
        worker_spec["worker_results_path"] = worker_results_path
        worker_spec["worker_infer_path"] = worker_infer_path
    return worker_specs


def _normalize_light_handler_inputs(handler_inputs=None):
    normalized = dict(handler_inputs or {})
    normalized.setdefault("save_cube_info", False)
    normalized.setdefault("n_save_cube_items", 0)
    normalized.setdefault("temp_cube_file", None)
    normalized.setdefault("summary_filename", None)
    normalized.setdefault("gen_esp_cube_flag", False)
    normalized.setdefault("keep_aux_files", False)
    return normalized


def _merge_dm_infer_results(
    worker_specs,
    final_results_dir,
    source_ase_db_path=None,
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    if os.path.exists(final_results_dir):
        shutil.rmtree(final_results_dir)
    os.makedirs(final_results_dir, exist_ok=True)

    merged_summary = []
    copied_dirs = 0

    for worker_spec in worker_specs:
        worker_results_dir = worker_spec["worker_results_path"]
        for local_idx, global_idx in enumerate(worker_spec["local_to_global"]):
            src_dir = os.path.join(worker_results_dir, str(local_idx))
            if not os.path.exists(src_dir):
                continue

            dst_dir = os.path.join(final_results_dir, str(global_idx))
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            copied_dirs += 1

            result_json = os.path.join(dst_dir, "dm_inference_result.json")
            if os.path.exists(result_json):
                with open(result_json, "r", encoding="utf-8") as f_obj:
                    item = json.load(f_obj)
                file_item = dict(item)
                file_item["idx"] = int(global_idx)
                with open(result_json, "w", encoding="utf-8") as f_obj:
                    json.dump(file_item, f_obj, indent=4)

                merged_item = dict(file_item)
                merged_item["source_idx"] = int(global_idx)
                merged_item["worker_id"] = int(worker_spec["worker_id"])
                merged_summary.append(merged_item)

    merged_summary.sort(key=lambda item: item.get("source_idx", -1))
    merged_summary_path = write_inference_summary_json(
        summary_data_list=merged_summary,
        results_folder_path=final_results_dir,
        summary_json_name=summary_json_name,
    )
    summary_npz_path = save_inference_summary_npz(
        summary_data_list=merged_summary,
        results_folder_path=final_results_dir,
        summary_filename=summary_filename,
    )
    resolved_updated_ase_db_path = None
    if updated_ase_db_path not in (None, False):
        if updated_ase_db_path == "auto":
            resolved_updated_ase_db_path = os.path.join(final_results_dir, DEFAULT_UPDATED_ASE_DB_NAME)
        else:
            resolved_updated_ase_db_path = os.path.abspath(updated_ase_db_path)
    if merged_summary and source_ase_db_path and resolved_updated_ase_db_path:
        write_dm_inference_ase_db(
            src_ase_path=source_ase_db_path,
            summary_data_list=merged_summary,
            dump_ase_db_path=resolved_updated_ase_db_path,
        )
    return {
        "copied_result_dirs": copied_dirs,
        "merged_items": len(merged_summary),
        "merged_summary_path": merged_summary_path,
        "summary_npz_path": summary_npz_path,
        "updated_ase_db_path": resolved_updated_ase_db_path,
    }


def _collect_handler_outputs(worker_specs):
    handler_outputs = []
    for worker_spec in worker_specs:
        output_path = os.path.join(worker_spec["worker_root"], "handler_outputs.json")
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Missing handler_outputs.json for worker {worker_spec['worker_name']}: {output_path}"
            )
        with open(output_path, "r", encoding="utf-8") as f_obj:
            payload = json.load(f_obj)
        attempted = int(payload.get("attempted", payload.get("processed", 0)))
        expected = int(worker_spec["num_items"])
        if attempted != expected:
            raise RuntimeError(
                f"Worker {worker_spec['worker_name']} attempted {attempted} items, expected {expected}"
            )
        handler_outputs.append(payload)
    return handler_outputs


def _build_handler_command(python_cmd):
    return f"{python_cmd} -m emoles.hpc.dm_infer --input-json handler_inputs.json 2>&1"


def local_dm_infer(
    n_parallel_job,
    n_cpu_per_job,
    main_db_path,
    infer_lmdb_path,
    python_cmd="python",
    handler_inputs=None,
    cooking_dir="cooking",
    reset=True,
    merge_results=True,
    final_results_dir=None,
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    main_db_path = os.path.abspath(main_db_path)
    infer_lmdb_path = os.path.abspath(infer_lmdb_path)
    cooking_path = prepare_cooking_dir(dirname=cooking_dir, reset=reset)

    handler_inputs = dict(handler_inputs or {})
    handler_inputs.setdefault("cpu_threads_per_worker", int(n_cpu_per_job))
    worker_specs = _prepare_dm_infer_shards(
        main_db_path=main_db_path,
        infer_lmdb_path=infer_lmdb_path,
        cooking_path=cooking_path,
        n_workers=int(n_parallel_job),
        handler_inputs=handler_inputs,
    )

    if not worker_specs:
        return None

    job_folders = [worker_spec["worker_root"] for worker_spec in worker_specs]
    job_queue, active_jobs = launch_local_job_queue(
        job_folders=job_folders,
        n_parallel_jobs=int(n_parallel_job),
        cmd_line=_build_handler_command(python_cmd),
    )
    wait_for_local_jobs(job_queue, active_jobs, poll_interval=10)
    handler_outputs = _collect_handler_outputs(worker_specs)

    if merge_results:
        if final_results_dir is None:
            final_results_dir = _default_results_dir(infer_lmdb_path)
        merged = _merge_dm_infer_results(
            worker_specs,
            os.path.abspath(final_results_dir),
            source_ase_db_path=main_db_path,
            summary_filename=summary_filename,
            summary_json_name=summary_json_name,
            updated_ase_db_path=updated_ase_db_path,
        )
        merged["worker_outputs"] = handler_outputs
        return merged
    return {"worker_roots": job_folders, "worker_outputs": handler_outputs}


def remote_dm_infer(
    n_parallel_machines,
    main_db_path,
    infer_lmdb_path,
    resrc_info,
    machine_info,
    handler_inputs=None,
    cooking_dir="cooking",
    reset=True,
    merge_results=True,
    final_results_dir=None,
    python_cmd="python",
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    main_db_path = os.path.abspath(main_db_path)
    infer_lmdb_path = os.path.abspath(infer_lmdb_path)
    cooking_path = prepare_cooking_dir(dirname=cooking_dir, reset=reset)

    worker_specs = _prepare_dm_infer_shards(
        main_db_path=main_db_path,
        infer_lmdb_path=infer_lmdb_path,
        cooking_path=cooking_path,
        n_workers=int(n_parallel_machines),
        handler_inputs=handler_inputs,
    )

    task_list = []
    for worker_spec in worker_specs:
        task_list.append(
            Task(
                command=_build_handler_command(python_cmd),
                task_work_path=f"{worker_spec['worker_name']}/",
                forward_files=[f"{worker_spec['worker_root']}/*"],
                backward_files=["results", "handler_outputs.json"],
            )
        )

    build_submission(cooking_path, machine_info, resrc_info, task_list)
    handler_outputs = _collect_handler_outputs(worker_specs)

    if merge_results:
        if final_results_dir is None:
            final_results_dir = _default_results_dir(infer_lmdb_path)
        merged = _merge_dm_infer_results(
            worker_specs,
            os.path.abspath(final_results_dir),
            source_ase_db_path=main_db_path,
            summary_filename=summary_filename,
            summary_json_name=summary_json_name,
            updated_ase_db_path=updated_ase_db_path,
        )
        merged["worker_outputs"] = handler_outputs
        return merged
    return {
        "worker_roots": [worker_spec["worker_root"] for worker_spec in worker_specs],
        "worker_outputs": handler_outputs,
    }


def local_dm_infer_light(
    n_parallel_job,
    n_cpu_per_job,
    main_db_path,
    infer_lmdb_path,
    python_cmd="python",
    handler_inputs=None,
    cooking_dir="cooking",
    reset=True,
    merge_results=True,
    final_results_dir=None,
    summary_filename=None,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return local_dm_infer(
        n_parallel_job=n_parallel_job,
        n_cpu_per_job=n_cpu_per_job,
        main_db_path=main_db_path,
        infer_lmdb_path=infer_lmdb_path,
        python_cmd=python_cmd,
        handler_inputs=_normalize_light_handler_inputs(handler_inputs),
        cooking_dir=cooking_dir,
        reset=reset,
        merge_results=merge_results,
        final_results_dir=final_results_dir,
        summary_filename=summary_filename,
        summary_json_name=summary_json_name,
        updated_ase_db_path=updated_ase_db_path,
    )


def remote_dm_infer_light(
    n_parallel_machines,
    main_db_path,
    infer_lmdb_path,
    resrc_info,
    machine_info,
    handler_inputs=None,
    cooking_dir="cooking",
    reset=True,
    merge_results=True,
    final_results_dir=None,
    python_cmd="python",
    summary_filename=None,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return remote_dm_infer(
        n_parallel_machines=n_parallel_machines,
        main_db_path=main_db_path,
        infer_lmdb_path=infer_lmdb_path,
        resrc_info=resrc_info,
        machine_info=machine_info,
        handler_inputs=_normalize_light_handler_inputs(handler_inputs),
        cooking_dir=cooking_dir,
        reset=reset,
        merge_results=merge_results,
        final_results_dir=final_results_dir,
        python_cmd=python_cmd,
        summary_filename=summary_filename,
        summary_json_name=summary_json_name,
        updated_ase_db_path=updated_ase_db_path,
    )


def run_handler_from_payload(payload):
    payload = dict(payload)
    cpu_threads_per_worker = payload.pop("cpu_threads_per_worker", None)
    max_items = payload.get("max_items", None)
    total_rows = _count_ase_db_rows(payload["abs_ase_path"])
    attempted = total_rows if max_items is None else min(int(max_items), total_rows)
    configure_worker_env(
        gpu_id=None,
        cpu_threads_per_worker=cpu_threads_per_worker,
    )
    summary_data = dm_infer_entry_from_lmdb(**payload)
    return {
        "attempted": int(attempted),
        "processed": int(attempted),
        "success_count": len(summary_data),
        "results_folder_path": os.path.abspath(payload["results_folder_path"]),
        "infer_lmdb_path": os.path.abspath(payload["infer_lmdb_path"]),
        "abs_ase_path": os.path.abspath(payload["abs_ase_path"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Shard handler for dm_infer_entry_from_lmdb.")
    parser.add_argument(
        "--input-json",
        default="handler_inputs.json",
        help="JSON payload for dm_infer_entry_from_lmdb.",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f_obj:
        payload = json.load(f_obj)

    output = run_handler_from_payload(payload)
    write_json_file(os.path.join(os.getcwd(), "handler_outputs.json"), output)


if __name__ == "__main__":
    main()
