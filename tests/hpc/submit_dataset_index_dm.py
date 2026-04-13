import json
import os
import sys
from pathlib import Path

sys.path.append("./")

from emoles.hpc import (
    build_bohrium_machine_info,
    build_dataset_dm_handler_inputs,
    build_default_cpu_resources,
    build_local_machine_info,
    submit_dataset_index_dm_infer_jobs,
)


WORKSPACE_ROOT = Path(
    os.getenv(
        "EMOLES_0413_WORKSPACE",
        "/share/mp_20_abacus_production/qh9_data/0413_infer_workspace",
    )
).resolve()
BACKEND = os.getenv("EMOLES_0413_BACKEND", "bohrium").strip().lower()
IMAGE_NAME = os.getenv(
    "EMOLES_0413_IMAGE",
    "registry.dp.tech/dptech/dp/native/prod-11729/dptb:0412-mk",
).strip()
CPU_PER_NODE = int(os.getenv("EMOLES_0413_CPU_PER_NODE", "32"))
N_SHARDS = int(os.getenv("EMOLES_0413_N_SHARDS", "1"))
RANDOM_SEED = int(
    os.getenv("EMOLES_0413_RANDOM_SEED", os.getenv("EMOLES_0413_SHUFFLE_SEED", "20260413"))
)
ITEMS_PER_SHARD = os.getenv("EMOLES_0413_ITEMS_PER_SHARD", "100")
ITEMS_PER_SHARD = None if ITEMS_PER_SHARD in (None, "") else int(ITEMS_PER_SHARD)
MAX_SOURCE_ROWS = os.getenv("EMOLES_0413_MAX_SOURCE_ROWS")
MAX_SOURCE_ROWS = None if MAX_SOURCE_ROWS in (None, "") else int(MAX_SOURCE_ROWS)


def _env_or_default(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


DATASETS = {
    "cho": {
        "dataset_path": _env_or_default(
            "EMOLES_0413_CHO_DATASET_PATH",
            "/bohr/emoles-dptb-infer-out-cho-0412-0l2t/v1",
        ),
        "remote_bundle_root": _env_or_default(
            "EMOLES_0413_CHO_BUNDLE_ROOT",
            "/bohr/emoles-dptb-infer-out-cho-0412-0l2t/v1/emoles_dptb_infer_out_cho_0412",
        ),
        "job_name": _env_or_default("EMOLES_0413_CHO_JOB_NAME", "emoles-dm-cho-index-0413"),
    },
    "cho_li": {
        "dataset_path": _env_or_default(
            "EMOLES_0413_CHO_LI_DATASET_PATH",
            "/bohr/emoles-dptb-infer-out-li-0412-kr2r/v1",
        ),
        "remote_bundle_root": _env_or_default(
            "EMOLES_0413_CHO_LI_BUNDLE_ROOT",
            "/bohr/emoles-dptb-infer-out-li-0412-kr2r/v1/emoles_dptb_infer_out_li_0412",
        ),
        "job_name": _env_or_default("EMOLES_0413_CHO_LI_JOB_NAME", "emoles-dm-cho-li-index-0413"),
    },
}


def _bool_env(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _selected_datasets():
    raw = os.getenv("EMOLES_0413_DATASETS", "cho,cho_li")
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    return [name for name in selected if name in DATASETS]


def _handler_inputs():
    return build_dataset_dm_handler_inputs(
        calc_esp_flag=_bool_env("EMOLES_0413_CALC_ESP", True),
        calc_electronic_flag=_bool_env("EMOLES_0413_CALC_ELECTRONIC", True),
        transform_dm_flag=_bool_env("EMOLES_0413_TRANSFORM_DM", True),
        unified_pcm_flag=_bool_env("EMOLES_0413_UNIFIED_PCM", True),
        summary_json_name=os.getenv("EMOLES_0413_SUMMARY_JSON", "merged_inference_summary.json"),
        updated_ase_db_path=os.getenv("EMOLES_0413_UPDATED_ASE_DB", "auto"),
    )


def _python_bin():
    if BACKEND == "local":
        return os.getenv("EMOLES_0413_PYTHON_BIN", "python")
    return os.getenv("EMOLES_0413_PYTHON_BIN", "/opt/mamba/bin/python")


def _machine_info(dataset_name, dataset_cfg):
    if BACKEND == "local":
        return build_local_machine_info(
            local_root=WORKSPACE_ROOT / "local_root",
            remote_root=WORKSPACE_ROOT / "remote_root",
        )

    return build_bohrium_machine_info(
        dataset_path=dataset_cfg["dataset_path"],
        local_root=WORKSPACE_ROOT / dataset_name,
        remote_root=f"/root/{WORKSPACE_ROOT.name}",
        job_name=dataset_cfg["job_name"],
        image_name=IMAGE_NAME,
        email=os.getenv("BOHRIUM_EMAIL"),
        password=os.getenv("BOHRIUM_PASSWORD"),
        project_id=os.getenv("BOHRIUM_PROJECT_ID"),
    )


def submit_dataset(dataset_name):
    dataset_cfg = DATASETS[dataset_name]
    payload = submit_dataset_index_dm_infer_jobs(
        dataset_name=dataset_name,
        remote_bundle_root=dataset_cfg["remote_bundle_root"],
        work_root=WORKSPACE_ROOT,
        machine_info=_machine_info(dataset_name, dataset_cfg),
        resrc_info=build_default_cpu_resources(cpu_per_node=CPU_PER_NODE),
        n_shards=N_SHARDS,
        random_seed=RANDOM_SEED,
        items_per_shard=ITEMS_PER_SHARD,
        max_source_rows=MAX_SOURCE_ROWS,
        handler_inputs=_handler_inputs(),
        python_bin=_python_bin(),
        exit_on_submit=(BACKEND != "local"),
        clean=False,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    for dataset_name in _selected_datasets():
        submit_dataset(dataset_name)
