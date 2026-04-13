import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "emoles"


def _import_from_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def _compile_source(path: Path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_refactor_modules_compile():
    critical_files = [
        SRC_ROOT / "electronic.py",
        SRC_ROOT / "evaluation.py",
        SRC_ROOT / "utils" / "__init__.py",
        SRC_ROOT / "utils" / "parallel.py",
        SRC_ROOT / "utils" / "filesystem.py",
        SRC_ROOT / "utils" / "matrix.py",
        SRC_ROOT / "utils" / "numbers.py",
        SRC_ROOT / "inference" / "chemistry.py",
        SRC_ROOT / "inference" / "parallel.py",
        SRC_ROOT / "inference" / "model_io.py",
        SRC_ROOT / "inference" / "postprocess.py",
        SRC_ROOT / "inference" / "common_tools.py",
        SRC_ROOT / "inference" / "infer_entry.py",
        SRC_ROOT / "loss.py",
        SRC_ROOT / "hpc" / "__init__.py",
        SRC_ROOT / "hpc" / "dm_infer.py",
        SRC_ROOT / "hpc" / "utils.py",
        SRC_ROOT / "hpc" / "gaussian.py",
        SRC_ROOT / "hpc" / "xtb.py",
        SRC_ROOT / "build" / "uma_core.py",
        SRC_ROOT / "build" / "uma_serial.py",
        SRC_ROOT / "build" / "uma_parallel_utils.py",
        SRC_ROOT / "build" / "uma_parallel.py",
        SRC_ROOT / "build" / "uma_entry.py",
        SRC_ROOT / "build" / "uma_entry_pll.py",
    ]

    for path in critical_files:
        _compile_source(path)


def test_wrapper_modules_point_to_new_locations():
    expectations = {
        SRC_ROOT / "inference" / "infer_entry.py": {
            "emoles.inference.model_io",
            "emoles.inference.common_tools",
            "emoles.inference.parallel",
            "emoles.inference",
        },
        SRC_ROOT / "inference" / "common_tools.py": {
            "emoles.inference",
            "emoles",
        },
        SRC_ROOT / "loss.py": {
            "emoles.electronic",
            "emoles.evaluation",
        },
        SRC_ROOT / "gau_parallel" / "gaussian_dpdispatcher.py": {
            "emoles.hpc.gaussian",
        },
        SRC_ROOT / "gau_parallel" / "gaussian_dpdispatcher_re_submit.py": {
            "emoles.hpc.gaussian",
        },
        SRC_ROOT / "xtb_parallel" / "xtb_dpdispatcher.py": {
            "emoles.hpc.xtb",
        },
        SRC_ROOT / "hpc" / "__init__.py": {
            "emoles.hpc.dm_infer",
            "emoles.hpc.gaussian",
            "emoles.hpc.xtb",
        },
        SRC_ROOT / "build" / "uma_entry.py": {
            "emoles.build.uma_core",
            "emoles.build.uma_serial",
        },
        SRC_ROOT / "build" / "uma_entry_pll.py": {
            "emoles.build.uma_core",
            "emoles.build.uma_parallel",
        },
        SRC_ROOT / "build" / "uma_parallel_utils.py": {
            "emoles.utils.parallel",
        },
    }

    for path, expected_modules in expectations.items():
        imported_modules = _import_from_modules(path)
        assert expected_modules.issubset(imported_modules), path

    infer_entry_source = (SRC_ROOT / "inference" / "infer_entry.py").read_text(encoding="utf-8")
    assert "from emoles.inference import postprocess" in infer_entry_source

    common_tools_source = (SRC_ROOT / "inference" / "common_tools.py").read_text(encoding="utf-8")
    assert "from emoles import electronic" in common_tools_source
    assert "from emoles.inference import chemistry" in common_tools_source
