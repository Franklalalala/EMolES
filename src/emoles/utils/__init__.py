from .filesystem import setup_db_path, setup_output_directory
from .matrix import (
    cut_and_cal_matrix,
    cut_matrix,
    generate_molecule_transform_indices,
    get_atom_in_mo_indices,
    get_shifted_ham,
    matrix_transform,
)
from .numbers import format_number, get_mo_occ, vec_cosine_similarity
from .parallel import (
    build_worker_gpu_plan,
    configure_worker_env,
    query_gpu_mem_mb,
    run_ramped_slot_pool,
    run_worker_pool,
    safe_log,
    set_thread_env,
    write_json_file,
)

try:
    from .db import (
        extract_lmdb_records,
        get_pickle_record_any,
        prepare_ase_db_worker_assignments,
        prepare_ase_db_worker_shards,
        resolve_lmdb_paths,
        update_ase_db_w_lmdb,
    )
except ModuleNotFoundError as exc:
    def extract_lmdb_records(*args, **kwargs):
        raise ModuleNotFoundError(
            "extract_lmdb_records requires optional ASE/LMDB dependencies"
        ) from exc

    def get_pickle_record_any(*args, **kwargs):
        raise ModuleNotFoundError(
            "get_pickle_record_any requires optional ASE/LMDB dependencies"
        ) from exc

    def prepare_ase_db_worker_assignments(*args, **kwargs):
        raise ModuleNotFoundError(
            "prepare_ase_db_worker_assignments requires optional ASE/LMDB dependencies"
        ) from exc

    def prepare_ase_db_worker_shards(*args, **kwargs):
        raise ModuleNotFoundError(
            "prepare_ase_db_worker_shards requires optional ASE/LMDB dependencies"
        ) from exc

    def resolve_lmdb_paths(*args, **kwargs):
        raise ModuleNotFoundError(
            "resolve_lmdb_paths requires optional ASE/LMDB dependencies"
        ) from exc

    def update_ase_db_w_lmdb(*args, **kwargs):
        raise ModuleNotFoundError(
            "update_ase_db_w_lmdb requires optional ASE/LMDB dependencies"
        ) from exc

__all__ = [
    "cut_and_cal_matrix",
    "cut_matrix",
    "format_number",
    "generate_molecule_transform_indices",
    "get_atom_in_mo_indices",
    "get_mo_occ",
    "get_shifted_ham",
    "matrix_transform",
    "build_worker_gpu_plan",
    "configure_worker_env",
    "extract_lmdb_records",
    "prepare_ase_db_worker_assignments",
    "prepare_ase_db_worker_shards",
    "get_pickle_record_any",
    "query_gpu_mem_mb",
    "run_ramped_slot_pool",
    "resolve_lmdb_paths",
    "run_worker_pool",
    "safe_log",
    "set_thread_env",
    "setup_db_path",
    "setup_output_directory",
    "update_ase_db_w_lmdb",
    "vec_cosine_similarity",
    "write_json_file",
]
