from emoles.inference.model_io import (
    ase_db_2_dummy_dptb_lmdb,
    default_fine_tune_ckpt_path,
    default_fine_tune_input_json_path,
    dptb_infer_from_ase_db,
    dptb_infer_to_lmdb_from_ase_db,
    merge_infer_lmdb_shards,
    save_info_2_lmdb,
    save_info_2_npy,
)
from emoles.inference.parallel import dptb_infer_to_lmdb_from_ase_db_pll
from emoles.inference.common_tools import load_npy_safe as _load_npy_safe


_POSTPROCESS_EXPORTS = {
    "_calc_energy_and_properties",
    "dm_infer_entry",
    "dm_infer_entry_from_lmdb",
    "dm_infer_light_entry",
    "dm_infer_light_entry_from_lmdb",
    "dm_infer_lightning_entry",
    "dm_infer_lightning_entry_from_lmdb",
    "get_dm_info_from_npy",
    "get_ham_info_from_npy",
    "write_dm_inference_ase_db",
    "write_inference_summary_json",
}


def __getattr__(name):
    if name in _POSTPROCESS_EXPORTS:
        from emoles.inference import postprocess

        return getattr(postprocess, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "_calc_energy_and_properties",
    "_load_npy_safe",
    "ase_db_2_dummy_dptb_lmdb",
    "default_fine_tune_ckpt_path",
    "default_fine_tune_input_json_path",
    "dm_infer_entry",
    "dm_infer_entry_from_lmdb",
    "dm_infer_light_entry",
    "dm_infer_light_entry_from_lmdb",
    "dm_infer_lightning_entry",
    "dm_infer_lightning_entry_from_lmdb",
    "dptb_infer_from_ase_db",
    "dptb_infer_to_lmdb_from_ase_db",
    "dptb_infer_to_lmdb_from_ase_db_pll",
    "get_dm_info_from_npy",
    "get_ham_info_from_npy",
    "merge_infer_lmdb_shards",
    "save_info_2_lmdb",
    "save_info_2_npy",
    "write_dm_inference_ase_db",
    "write_inference_summary_json",
]
