import json
import os
import sys
from pathlib import Path

sys.path.append("./")

from emoles.hpc import aggregate_dataset_index_dm_results


WORKSPACE_ROOT = Path(
    os.getenv(
        "EMOLES_0413_WORKSPACE",
        "/share/mp_20_abacus_production/qh9_data/0413_infer_workspace",
    )
).resolve()


def _selected_datasets():
    raw = os.getenv("EMOLES_0413_DATASETS", "cho,cho_li")
    return [item.strip() for item in raw.split(",") if item.strip()]


def aggregate_dataset(dataset_name):
    payload = aggregate_dataset_index_dm_results(WORKSPACE_ROOT / dataset_name)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    for dataset_name in _selected_datasets():
        aggregate_dataset(dataset_name)
