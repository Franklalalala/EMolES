import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_utils_package_preserves_legacy_imports():
    from emoles.utils import (
        build_worker_gpu_plan,
        cut_and_cal_matrix,
        feed_ase_db_task_queue,
        format_number,
        get_mo_occ,
        prepare_ase_db_worker_assignments,
        run_ase_db_task_queue_pool,
        run_ramped_slot_pool,
    )

    assert format_number(12.345) == "12.35"
    assert format_number(0.01234) == "0.0123"
    assert get_mo_occ(4, 2).tolist() == [2.0, 2.0, 0.0, 0.0]
    assert build_worker_gpu_plan(gpus=[0, 1], workers_per_gpu=2) == [0, 0, 1, 1]
    assert callable(prepare_ase_db_worker_assignments)
    assert callable(feed_ase_db_task_queue)
    assert callable(run_ase_db_task_queue_pool)
    assert callable(run_ramped_slot_pool)

    diag_mae, non_diag_mae = cut_and_cal_matrix(
        full_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        atom_in_mo_indices=[0, 1],
    )
    assert diag_mae == 2.5
    assert non_diag_mae == 2.5


def test_parallel_worker_env_defaults_to_highest_precision():
    parallel_source = (SRC_ROOT / "emoles" / "utils" / "parallel.py").read_text(encoding="utf-8")
    assert 'EMOLES_FLOAT32_MATMUL_PRECISION' in parallel_source
    assert '"highest"' in parallel_source
