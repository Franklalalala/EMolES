import json
import os
import shutil
from pathlib import Path

from ase.db import connect
from dpdispatcher import Task

from emoles.inference.postprocess import (
    DEFAULT_SUMMARY_JSON_NAME,
    DEFAULT_UPDATED_ASE_DB_NAME,
    dm_infer_light_entry_from_lmdb,
)
from emoles.hpc.utils import (
    _normalize_bohrium_remote_profile,
    _patch_dargs_allow_ref,
    build_submission,
)


TASK_NAME = "task_0000"
SUBMIT_DIRNAME = "submit"
REMOTE_RESULTS_DIRNAME = "remote_results"
DEFAULT_TASK_HANDLER = Path(__file__).resolve().with_name("dataset_index_dm_handler.py")
DEFAULT_HANDLER_INPUTS = {
    "matrix_field": "hamiltonian",
    "convention": "def2svp",
    "mol_charge": 0,
    "transform_dm_flag": True,
    "calc_esp_flag": True,
    "calc_electronic_flag": True,
    "unified_pcm_flag": True,
    "summary_json_name": DEFAULT_SUMMARY_JSON_NAME,
    "updated_ase_db_path": "auto",
}

_patch_dargs_allow_ref()


def resolve_dataset_bundle(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    if (dataset_root / "optimized_all.db").exists() and (dataset_root / "infer").exists():
        return dataset_root

    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "optimized_all.db").exists() and (child / "infer").exists():
            return child

    raise FileNotFoundError(f"Dataset bundle not found under: {dataset_root}")

def _extract_submission_summary(result):
    belonging_jobs = result.get("belonging_jobs", []) if isinstance(result, dict) else []
    if not belonging_jobs:
        return {
            "submission_hash": None,
            "job_id": None,
            "work_base": result.get("work_base") if isinstance(result, dict) else None,
        }

    first_group = belonging_jobs[0]
    if not first_group:
        return {
            "submission_hash": None,
            "job_id": None,
            "work_base": result.get("work_base") if isinstance(result, dict) else None,
        }

    submission_hash = next(iter(first_group.keys()))
    job_payload = first_group[submission_hash]
    return {
        "submission_hash": submission_hash,
        "job_id": job_payload.get("job_id"),
        "work_base": result.get("work_base") if isinstance(result, dict) else None,
    }


def run_remote_dataset_task(config_path="task_config.json"):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    dataset_root = resolve_dataset_bundle(config["dataset_root"])
    output_dir = Path(config.get("output_dir") or REMOTE_RESULTS_DIRNAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    handler_inputs = dict(config.get("handler_inputs") or {})
    handler_inputs.setdefault("summary_json_name", DEFAULT_SUMMARY_JSON_NAME)
    handler_inputs.setdefault(
        "updated_ase_db_path",
        str(output_dir / DEFAULT_UPDATED_ASE_DB_NAME),
    )

    try:
        summary_data = dm_infer_light_entry_from_lmdb(
            abs_ase_path=str(dataset_root / "optimized_all.db"),
            infer_lmdb_path=str(dataset_root / "infer"),
            results_folder_path=str(output_dir),
            **handler_inputs,
        )
        payload = {
            "success": True,
            "dataset_name": config.get("dataset_name"),
            "dataset_root": str(dataset_root),
            "updated_ase_db_path": handler_inputs["updated_ase_db_path"],
            "summary_json_path": str(output_dir / handler_inputs["summary_json_name"]),
            "item_count": len(summary_data),
        }
    except Exception as exc:
        payload = {
            "success": False,
            "dataset_name": config.get("dataset_name"),
            "dataset_root": str(dataset_root),
            "error": repr(exc),
        }
        (output_dir / "task_result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

    (output_dir / "task_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def submit_bohrium_dataset_dm_infer_job(
    dataset_name,
    dataset_root,
    work_root,
    machine_info,
    resrc_info,
    handler_inputs=None,
):
    work_root = Path(work_root).resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    submit_root = work_root / SUBMIT_DIRNAME
    task_dir = submit_root / TASK_NAME
    task_dir.mkdir(parents=True, exist_ok=True)

    task_config = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "output_dir": REMOTE_RESULTS_DIRNAME,
        "handler_inputs": dict(handler_inputs or {}),
    }
    (task_dir / "task_config.json").write_text(
        json.dumps(task_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_dir / "run.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "PYTHON_BIN=$(command -v python || command -v python3)",
                'echo "python=${PYTHON_BIN}"',
                "$PYTHON_BIN -c \"from emoles.hpc.bohrium_dataset_dm_infer import run_remote_dataset_task; run_remote_dataset_task('task_config.json')\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(task_dir / "run.sh", 0o755)

    task = Task(
        command="bash run.sh > task.stdout 2>&1",
        task_work_path=f"{TASK_NAME}/",
        forward_files=[f"{task_dir}/*"],
        backward_files=[REMOTE_RESULTS_DIRNAME, "task.stdout"],
    )
    result = build_submission(
        str(submit_root),
        {
            **dict(machine_info),
            "local_root": str(work_root),
            "remote_profile": _normalize_bohrium_remote_profile(machine_info["remote_profile"]),
        },
        dict(resrc_info),
        [task],
        run_submission_kwargs={"exit_on_submit": True, "clean": False},
    )
    summary = _extract_submission_summary(result)
    summary["dataset_name"] = dataset_name
    summary["dataset_root"] = dataset_root
    summary["work_root"] = str(work_root)
    (work_root / "submitted_job.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_bohrium_machine_info(
    dataset_path,
    local_root,
    job_name,
    image_name,
    *,
    remote_root=".",
    email=None,
    password=None,
    project_id=None,
    disk_size=200,
    scass_type="c32_m64_cpu",
    platform="ali",
    log_file="log",
):
    remote_profile = {
        "input_data": {
            "job_type": "container",
            "log_file": str(log_file),
            "job_name": str(job_name),
            "disk_size": int(disk_size),
            "scass_type": str(scass_type).strip(),
            "platform": str(platform).strip(),
            "image_name": str(image_name).strip(),
            "dataset_path": [str(dataset_path)],
        }
    }
    if email not in (None, ""):
        remote_profile["email"] = str(email)
    if password not in (None, ""):
        remote_profile["password"] = str(password)
    if project_id not in (None, ""):
        remote_profile["project_id"] = int(project_id)

    return {
        "batch_type": "Bohrium",
        "context_type": "Bohrium",
        "local_root": str(local_root),
        "remote_root": str(remote_root),
        "remote_profile": remote_profile,
    }


def build_local_machine_info(*, local_root, remote_root, batch_type="Shell", context_type="LocalContext"):
    return {
        "batch_type": str(batch_type),
        "context_type": str(context_type),
        "local_root": str(local_root),
        "remote_root": str(remote_root),
    }


def build_default_cpu_resources(
    *,
    cpu_per_node=32,
    queue_name="LBG_CPU",
    group_size=1,
    envs=None,
):
    return {
        "number_node": 1,
        "cpu_per_node": int(cpu_per_node),
        "gpu_per_node": 0,
        "group_size": int(group_size),
        "queue_name": str(queue_name),
        "envs": dict(envs or {"PYTHONUNBUFFERED": "1"}),
        "strategy": {"ratio_unfinished": 0.0},
    }


def build_dataset_dm_handler_inputs(**overrides):
    payload = dict(DEFAULT_HANDLER_INPUTS)
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return payload


def _submission_jobs(result):
    jobs = []
    if not isinstance(result, dict):
        return jobs
    for group in result.get("belonging_jobs", []):
        for submission_hash, payload in group.items():
            jobs.append(
                {
                    "submission_hash": submission_hash,
                    "job_id": payload.get("job_id"),
                }
            )
    return jobs


def _resolve_dataset_work_root(work_root, dataset_name):
    return Path(work_root).resolve() / str(dataset_name)


def _dataset_remote_root(machine_info, work_root, dataset_name):
    base_remote_root = machine_info.get("remote_root")
    if base_remote_root in (None, ""):
        base_remote_root = f"/root/{Path(work_root).resolve().name}"
    base_remote_root = str(base_remote_root)
    if Path(base_remote_root).name == str(dataset_name):
        return base_remote_root
    return os.path.join(base_remote_root, str(dataset_name))


def _write_dataset_index_task_config(
    task_dir,
    dataset_name,
    remote_bundle_root,
    shard_id,
    n_shards,
    random_seed,
    handler_inputs=None,
    items_per_shard=None,
    max_source_rows=None,
):
    payload = {
        "dataset_name": str(dataset_name),
        "remote_bundle_root": str(remote_bundle_root),
        "shard_id": int(shard_id),
        "n_shards": int(n_shards),
        "random_seed": int(random_seed),
    }
    payload.update(dict(handler_inputs or {}))
    if items_per_shard not in (None, ""):
        payload["items_per_shard"] = int(items_per_shard)
    if max_source_rows not in (None, ""):
        payload["max_source_rows"] = int(max_source_rows)

    (Path(task_dir) / "task_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def submit_dataset_index_dm_infer_jobs(
    *,
    dataset_name,
    remote_bundle_root,
    work_root,
    machine_info,
    resrc_info,
    n_shards=1,
    random_seed=20260413,
    items_per_shard=None,
    max_source_rows=None,
    handler_inputs=None,
    python_bin="/opt/mamba/bin/python",
    handler_script_path=None,
    exit_on_submit=True,
    clean=False,
):
    dataset_work_root = _resolve_dataset_work_root(work_root, dataset_name)
    if dataset_work_root.exists():
        shutil.rmtree(dataset_work_root)
    dataset_work_root.mkdir(parents=True, exist_ok=True)

    handler_script_path = Path(handler_script_path or DEFAULT_TASK_HANDLER).resolve()
    task_list = []
    for shard_id in range(int(n_shards)):
        task_dir = dataset_work_root / str(shard_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(handler_script_path, task_dir / handler_script_path.name)
        _write_dataset_index_task_config(
            task_dir=task_dir,
            dataset_name=dataset_name,
            remote_bundle_root=remote_bundle_root,
            shard_id=shard_id,
            n_shards=n_shards,
            random_seed=random_seed,
            handler_inputs=handler_inputs,
            items_per_shard=items_per_shard,
            max_source_rows=max_source_rows,
        )
        task_list.append(
            Task(
                command=f"{python_bin} {handler_script_path.name} > task.stdout 2>&1",
                task_work_path=f"{shard_id}/",
                forward_files=[f"{task_dir}/*"],
                backward_files=["results", "task.stdout"],
            )
        )

    machine_payload = dict(machine_info)
    machine_payload["local_root"] = str(dataset_work_root)
    machine_payload["remote_root"] = _dataset_remote_root(
        machine_info=machine_info,
        work_root=work_root,
        dataset_name=dataset_name,
    )
    if machine_payload.get("remote_profile"):
        machine_payload["remote_profile"] = _normalize_bohrium_remote_profile(
            machine_payload["remote_profile"]
        )

    result = build_submission(
        str(dataset_work_root),
        machine_payload,
        dict(resrc_info),
        task_list,
        run_submission_kwargs={"exit_on_submit": bool(exit_on_submit), "clean": bool(clean)},
    )
    payload = {
        "dataset_name": str(dataset_name),
        "n_shards": int(n_shards),
        "random_seed": int(random_seed),
        "items_per_shard": None if items_per_shard in (None, "") else int(items_per_shard),
        "max_source_rows": None if max_source_rows in (None, "") else int(max_source_rows),
        "remote_bundle_root": str(remote_bundle_root),
        "handler_script_path": str(handler_script_path),
        "jobs": _submission_jobs(result),
    }
    (dataset_work_root / "submitted_jobs.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _remove_sqlite_sidecars(db_path):
    for path in (db_path, f"{db_path}-shm", f"{db_path}-wal"):
        if os.path.exists(path):
            os.remove(path)


def _row_sort_key(row):
    data = dict(getattr(row, "data", None) or {})
    for key in ("source_idx", "dm_infer_source_idx", "dm_infer_idx"):
        value = data.get(key)
        if value is None:
            continue
        return int(value)
    return int(getattr(row, "id", 0))


def aggregate_dataset_index_dm_results(dataset_root, output_name=DEFAULT_UPDATED_ASE_DB_NAME):
    dataset_root = Path(dataset_root).resolve()
    shard_db_paths = []
    for shard_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()):
        shard_db_path = shard_dir / "results" / DEFAULT_UPDATED_ASE_DB_NAME
        if shard_db_path.exists():
            shard_db_paths.append(shard_db_path)

    rows = []
    for shard_db_path in shard_db_paths:
        with connect(str(shard_db_path)) as db:
            for row in db.select():
                rows.append(
                    {
                        "sort_key": _row_sort_key(row),
                        "atoms": row.toatoms(),
                        "data": dict(getattr(row, "data", None) or {}),
                        "kvp": dict(getattr(row, "key_value_pairs", None) or {}),
                    }
                )

    rows.sort(key=lambda item: item["sort_key"])
    output_db_path = dataset_root / output_name
    _remove_sqlite_sidecars(str(output_db_path))
    with connect(str(output_db_path)) as dump_db:
        for item in rows:
            dump_db.write(item["atoms"], key_value_pairs=item["kvp"], data=item["data"])

    payload = {
        "dataset_root": str(dataset_root),
        "n_rows": len(rows),
        "output_db_path": str(output_db_path),
        "shard_db_paths": [str(path) for path in shard_db_paths],
    }
    (dataset_root / "aggregate_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
