import importlib.util
import json
import sys
import types
from pathlib import Path

from ase import Atoms
from ase.db import connect


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "emoles" / "hpc" / "bohrium_dataset_dm_infer.py"


def _load_dataset_module(monkeypatch):
    fake_postprocess = types.ModuleType("emoles.inference.postprocess")
    fake_postprocess.DEFAULT_SUMMARY_JSON_NAME = "merged_inference_summary.json"
    fake_postprocess.DEFAULT_UPDATED_ASE_DB_NAME = "dm_inference_results.db"
    fake_postprocess.dm_infer_light_entry_from_lmdb = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "emoles.inference.postprocess", fake_postprocess)

    fake_utils = types.ModuleType("emoles.hpc.utils")
    fake_utils.build_submission = lambda *args, **kwargs: None
    fake_utils._normalize_bohrium_remote_profile = lambda payload: dict(payload)
    fake_utils._patch_dargs_allow_ref = lambda: None
    monkeypatch.setitem(sys.modules, "emoles.hpc.utils", fake_utils)

    class FakeArgument:
        def normalize_value(
            self,
            value,
            inplace=False,
            do_default=True,
            do_alias=True,
            trim_pattern=None,
        ):
            return value

    dargs_mod = types.ModuleType("dargs")
    dargs_mod.Argument = FakeArgument
    monkeypatch.setitem(sys.modules, "dargs", dargs_mod)

    class FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    dpdispatcher_mod = types.ModuleType("dpdispatcher")
    dpdispatcher_mod.Task = FakeTask
    dpdispatcher_mod.Submission = object
    monkeypatch.setitem(sys.modules, "dpdispatcher", dpdispatcher_mod)

    bohrium_context_mod = types.ModuleType("dpdispatcher.contexts.dp_cloud_server_context")
    bohrium_context_mod.BohriumContext = object
    monkeypatch.setitem(sys.modules, "dpdispatcher.contexts.dp_cloud_server_context", bohrium_context_mod)

    bohrium_machine_mod = types.ModuleType("dpdispatcher.machines.dp_cloud_server")
    bohrium_machine_mod.Bohrium = object
    monkeypatch.setitem(sys.modules, "dpdispatcher.machines.dp_cloud_server", bohrium_machine_mod)

    submission_mod = types.ModuleType("dpdispatcher.submission")
    submission_mod.Resources = object
    monkeypatch.setitem(sys.modules, "dpdispatcher.submission", submission_mod)

    module_name = "test_hpc_dataset_index_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_submit_dataset_index_dm_infer_jobs_writes_task_files(monkeypatch, tmp_path):
    module = _load_dataset_module(monkeypatch)
    build_calls = {}

    def fake_build_submission(cooking_path, machine_info, resrc_info, task_list, run_submission_kwargs=None):
        build_calls["cooking_path"] = cooking_path
        build_calls["machine_info"] = dict(machine_info)
        build_calls["resrc_info"] = dict(resrc_info)
        build_calls["task_list"] = list(task_list)
        build_calls["run_submission_kwargs"] = dict(run_submission_kwargs or {})
        return {
            "belonging_jobs": [
                {
                    "hash-a": {"job_id": "job-001"},
                    "hash-b": {"job_id": "job-002"},
                }
            ]
        }

    monkeypatch.setattr(module, "build_submission", fake_build_submission)

    handler_script = tmp_path / "dataset_index_dm_handler.py"
    handler_script.write_text("print('ok')\n", encoding="utf-8")
    work_root = tmp_path / "workspace"

    payload = module.submit_dataset_index_dm_infer_jobs(
        dataset_name="cho",
        remote_bundle_root="/bohr/example/cho_bundle",
        work_root=work_root,
        machine_info={
            "batch_type": "Shell",
            "context_type": "LocalContext",
            "local_root": str(tmp_path / "local_root"),
            "remote_root": str(tmp_path / "remote_root"),
        },
        resrc_info={
            "number_node": 1,
            "cpu_per_node": 1,
            "gpu_per_node": 0,
            "queue_name": "local",
            "group_size": 1,
            "envs": {},
            "strategy": {"ratio_unfinished": 0.0},
        },
        n_shards=2,
        random_seed=17,
        items_per_shard=3,
        handler_inputs={"calc_esp_flag": False},
        python_bin="/usr/bin/python",
        handler_script_path=handler_script,
        exit_on_submit=False,
        clean=False,
    )

    dataset_root = work_root / "cho"
    assert build_calls["cooking_path"] == str(dataset_root)
    assert build_calls["machine_info"]["local_root"] == str(dataset_root)
    assert build_calls["machine_info"]["remote_root"].endswith("remote_root\\cho") or build_calls["machine_info"]["remote_root"].endswith("remote_root/cho")
    assert build_calls["run_submission_kwargs"] == {"exit_on_submit": False, "clean": False}
    assert len(build_calls["task_list"]) == 2
    assert payload["jobs"] == [
        {"submission_hash": "hash-a", "job_id": "job-001"},
        {"submission_hash": "hash-b", "job_id": "job-002"},
    ]

    first_task = build_calls["task_list"][0]
    assert first_task.kwargs["command"] == "/usr/bin/python dataset_index_dm_handler.py > task.stdout 2>&1"
    assert first_task.kwargs["task_work_path"] == "0/"

    task_config = json.loads((dataset_root / "0" / "task_config.json").read_text(encoding="utf-8"))
    assert task_config["dataset_name"] == "cho"
    assert task_config["remote_bundle_root"] == "/bohr/example/cho_bundle"
    assert task_config["shard_id"] == 0
    assert task_config["n_shards"] == 2
    assert task_config["random_seed"] == 17
    assert task_config["items_per_shard"] == 3
    assert task_config["calc_esp_flag"] is False
    assert (dataset_root / "0" / "dataset_index_dm_handler.py").exists()
    assert json.loads((dataset_root / "submitted_jobs.json").read_text(encoding="utf-8"))["n_shards"] == 2


def test_aggregate_dataset_index_dm_results_sorts_rows_by_source_idx(monkeypatch, tmp_path):
    module = _load_dataset_module(monkeypatch)
    dataset_root = tmp_path / "cho"
    shard0 = dataset_root / "0" / "results"
    shard1 = dataset_root / "1" / "results"
    shard0.mkdir(parents=True, exist_ok=True)
    shard1.mkdir(parents=True, exist_ok=True)

    db_name = module.DEFAULT_UPDATED_ASE_DB_NAME
    with connect(str(shard0 / db_name)) as db:
        db.write(Atoms("H", positions=[[0.0, 0.0, 0.0]]), data={"source_idx": 2})
    with connect(str(shard1 / db_name)) as db:
        db.write(Atoms("He", positions=[[0.0, 0.0, 0.0]]), data={"source_idx": 0})
        db.write(Atoms("Li", positions=[[0.0, 0.0, 0.0]]), data={"source_idx": 1})

    payload = module.aggregate_dataset_index_dm_results(dataset_root)
    output_db = Path(payload["output_db_path"])
    assert payload["n_rows"] == 3
    assert output_db.exists()

    with connect(str(output_db)) as db:
        source_indices = [row.data["source_idx"] for row in db.select()]
        symbols = [row.toatoms().symbols[0] for row in db.select()]

    assert source_indices == [0, 1, 2]
    assert symbols == ["He", "Li", "H"]
