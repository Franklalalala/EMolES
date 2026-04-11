import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "emoles"


def _top_level_defs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _module_all(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return {
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }
    return set()


def _function_defaults(path: Path, function_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            arg_names = [arg.arg for arg in node.args.args]
            defaults = node.args.defaults
            default_offset = len(arg_names) - len(defaults)
            output = {}
            for idx, value in enumerate(defaults):
                output[arg_names[default_offset + idx]] = ast.unparse(value)
            return output
    return {}


def _compile_source(path: Path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_lmdb_infer_modules_compile():
    for path in (
        SRC_ROOT / "utils" / "db.py",
        SRC_ROOT / "utils" / "parallel.py",
        SRC_ROOT / "hpc" / "__init__.py",
        SRC_ROOT / "hpc" / "dm_infer.py",
        SRC_ROOT / "hpc" / "utils.py",
        SRC_ROOT / "inference" / "model_io.py",
        SRC_ROOT / "inference" / "parallel.py",
        SRC_ROOT / "inference" / "infer_entry.py",
        SRC_ROOT / "inference" / "postprocess.py",
    ):
        _compile_source(path)


def test_lmdb_infer_api_is_exposed():
    model_io_defs = _top_level_defs(SRC_ROOT / "inference" / "model_io.py")
    assert "dptb_infer_to_lmdb_from_ase_db" in model_io_defs
    assert "merge_infer_lmdb_shards" in model_io_defs

    pll_defs = _top_level_defs(SRC_ROOT / "inference" / "parallel.py")
    assert "dptb_infer_to_lmdb_from_ase_db_pll" in pll_defs

    postprocess_defs = _top_level_defs(SRC_ROOT / "inference" / "postprocess.py")
    assert "dm_infer_light_entry" in postprocess_defs
    assert "dm_infer_light_entry_from_lmdb" in postprocess_defs
    assert "dm_infer_lightning_entry" in postprocess_defs
    assert "dm_infer_lightning_entry_from_lmdb" in postprocess_defs
    assert "write_dm_inference_ase_db" in postprocess_defs

    hpc_defs = _top_level_defs(SRC_ROOT / "hpc" / "dm_infer.py")
    assert "local_dm_infer_light" in hpc_defs
    assert "remote_dm_infer_light" in hpc_defs

    exported_names = _module_all(SRC_ROOT / "inference" / "infer_entry.py")
    assert "dptb_infer_to_lmdb_from_ase_db" in exported_names
    assert "dptb_infer_to_lmdb_from_ase_db_pll" in exported_names
    assert "dm_infer_entry_from_lmdb" in exported_names
    assert "dm_infer_light_entry" in exported_names
    assert "dm_infer_light_entry_from_lmdb" in exported_names
    assert "dm_infer_lightning_entry" in exported_names
    assert "dm_infer_lightning_entry_from_lmdb" in exported_names
    assert "write_dm_inference_ase_db" in exported_names
    assert "merge_infer_lmdb_shards" in exported_names


def test_lmdb_infer_defaults_and_manifest_contract():
    model_io_path = SRC_ROOT / "inference" / "model_io.py"
    db_utils_path = SRC_ROOT / "utils" / "db.py"
    parallel_path = SRC_ROOT / "inference" / "parallel.py"

    direct_defaults = _function_defaults(model_io_path, "dptb_infer_from_ase_db")
    assert direct_defaults.get("max_items") == "None"

    lmdb_defaults = _function_defaults(model_io_path, "dptb_infer_to_lmdb_from_ase_db")
    assert lmdb_defaults.get("max_items") == "None"

    model_io_source = model_io_path.read_text(encoding="utf-8")
    assert 'os.path.join(infer_root, "manifest.json")' in model_io_source

    db_utils_source = db_utils_path.read_text(encoding="utf-8")
    assert 'manifest_path = os.path.join(abs_path, "manifest.json")' in db_utils_source
    assert 'manifest.get("worker_lmdb_paths", [])' in db_utils_source

    parallel_source = parallel_path.read_text(encoding="utf-8")
    assert "shutil.rmtree(infer_root)" in parallel_source

    postprocess_source = (SRC_ROOT / "inference" / "postprocess.py").read_text(encoding="utf-8")
    assert 'DEFAULT_UPDATED_ASE_DB_NAME = "dm_inference_results.db"' in postprocess_source
    assert "write_dm_inference_ase_db(" in postprocess_source
    assert "keep_aux_files=False" in postprocess_source
    assert '"ESP_max_eV"] = _optional_float(esp_max)' in postprocess_source

    hpc_source = (SRC_ROOT / "hpc" / "dm_infer.py").read_text(encoding="utf-8")
    assert '"attempted": int(attempted)' in hpc_source
    assert '"success_count": len(summary_data)' in hpc_source
    assert 'file_item["idx"] = int(global_idx)' in hpc_source
    assert "def remote_dm_infer_light(" in hpc_source

    hpc_utils_source = (SRC_ROOT / "hpc" / "utils.py").read_text(encoding="utf-8")
    assert 'if context_type == "LocalContext":' in hpc_utils_source
    assert 'run_tag = f"run_{int(time.time() * 1000)}"' in hpc_utils_source
