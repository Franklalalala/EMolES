import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "src" / "emoles" / "hpc" / "utils.py"


def _load_hpc_utils(monkeypatch):
    state = {}

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

        def check_value(self, value, strict=False):
            return value

    dargs_mod = types.ModuleType("dargs")
    dargs_mod.Argument = FakeArgument
    monkeypatch.setitem(sys.modules, "dargs", dargs_mod)

    class FakeMachine:
        load_calls = []

        @classmethod
        def load_from_dict(cls, machine_dict):
            FakeArgument().check_value(machine_dict, strict=False, allow_ref=False)
            cls.load_calls.append(dict(machine_dict))
            return {"kind": "loaded-machine", "machine_dict": dict(machine_dict)}

    class FakeResources:
        load_calls = []
        init_calls = []

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            type(self).init_calls.append(dict(kwargs))

        @classmethod
        def load_from_dict(cls, resources_dict):
            cls.load_calls.append(dict(resources_dict))
            raise TypeError("Argument.normalize_value() got an unexpected keyword argument 'allow_ref'")

    class FakeSubmission:
        last_instance = None

        def __init__(self, work_base, machine, resources, task_list):
            self.work_base = work_base
            self.machine = machine
            self.resources = resources
            self.task_list = list(task_list)
            self.run_calls = 0
            type(self).last_instance = self

        def run_submission(self):
            self.run_calls += 1

    class FakeBohriumContext:
        def __init__(self, local_root, remote_root=None, remote_profile=None, *args, **kwargs):
            self.local_root = local_root
            self.remote_root = remote_root
            self.remote_profile = dict(remote_profile or {})

    class FakeBohrium:
        init_calls = []

        def __init__(self, context, **kwargs):
            self.context = context
            self.kwargs = dict(kwargs)
            type(self).init_calls.append({"context": context, "kwargs": dict(kwargs)})

    dpdispatcher_mod = types.ModuleType("dpdispatcher")
    dpdispatcher_mod.Machine = FakeMachine
    dpdispatcher_mod.Resources = FakeResources
    dpdispatcher_mod.Submission = FakeSubmission
    monkeypatch.setitem(sys.modules, "dpdispatcher", dpdispatcher_mod)

    contexts_pkg = types.ModuleType("dpdispatcher.contexts")
    bohrium_context_mod = types.ModuleType("dpdispatcher.contexts.dp_cloud_server_context")
    bohrium_context_mod.BohriumContext = FakeBohriumContext
    monkeypatch.setitem(sys.modules, "dpdispatcher.contexts", contexts_pkg)
    monkeypatch.setitem(sys.modules, "dpdispatcher.contexts.dp_cloud_server_context", bohrium_context_mod)

    machines_pkg = types.ModuleType("dpdispatcher.machines")
    bohrium_machine_mod = types.ModuleType("dpdispatcher.machines.dp_cloud_server")
    bohrium_machine_mod.Bohrium = FakeBohrium
    monkeypatch.setitem(sys.modules, "dpdispatcher.machines", machines_pkg)
    monkeypatch.setitem(sys.modules, "dpdispatcher.machines.dp_cloud_server", bohrium_machine_mod)

    module_name = "test_hpc_utils_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    state["argument_cls"] = FakeArgument
    state["machine_cls"] = FakeMachine
    state["resources_cls"] = FakeResources
    state["submission_cls"] = FakeSubmission
    state["bohrium_cls"] = FakeBohrium
    return module, state


def test_build_submission_uses_direct_bohrium_machine(monkeypatch, tmp_path):
    utils, state = _load_hpc_utils(monkeypatch)

    machine_info = {
        "batch_type": "Bohrium",
        "context_type": "BohriumContext",
        "local_root": str(tmp_path / "bohrium_root"),
        "remote_root": ".",
        "remote_profile": {
            "email": "user@example.com",
            "password": "secret",
            "project_id": "320004",
            "input_data": {
                "job_type": " container ",
                "log_file": " log ",
                "job_name": " test-job ",
                "disk_size": "200",
                "scass_type": "c32_m64_cpu ",
                "platform": " ali ",
                "image_name": " image:tag ",
            },
        },
        "retry_count": 3,
    }
    resrc_info = {
        "number_node": 1,
        "cpu_per_node": 32,
        "gpu_per_node": 0,
        "queue_name": "LBG_CPU",
        "group_size": 1,
        "envs": {},
        "strategy": {"ratio_unfinished": 0.0},
    }

    utils.build_submission(str(tmp_path / "cook"), machine_info, resrc_info, task_list=["task"])

    submission = state["submission_cls"].last_instance
    assert isinstance(submission.machine, state["bohrium_cls"])
    assert submission.run_calls == 1
    assert state["machine_cls"].load_calls == []

    remote_profile = submission.machine.context.remote_profile
    assert remote_profile["project_id"] == 320004
    assert remote_profile["input_data"]["job_type"] == "container"
    assert remote_profile["input_data"]["log_file"] == "log"
    assert remote_profile["input_data"]["job_name"] == "test-job"
    assert remote_profile["input_data"]["disk_size"] == 200
    assert remote_profile["input_data"]["scass_type"] == "c32_m64_cpu"
    assert remote_profile["input_data"]["platform"] == "ali"
    assert remote_profile["input_data"]["image_name"] == "image:tag"

    assert state["resources_cls"].load_calls == [resrc_info]
    assert state["resources_cls"].init_calls[-1] == resrc_info
    assert "allow_ref" in inspect.signature(state["argument_cls"].normalize_value).parameters
    assert "allow_ref" in inspect.signature(state["argument_cls"].check_value).parameters


def test_build_submission_local_context_splits_same_roots(monkeypatch, tmp_path):
    utils, state = _load_hpc_utils(monkeypatch)

    shared_root = tmp_path / "shared"
    cooking_path = tmp_path / "cooking"
    machine_info = {
        "batch_type": "Shell",
        "context_type": "LocalContext",
        "local_root": str(shared_root),
        "remote_root": str(shared_root),
    }
    resrc_info = {
        "number_node": 1,
        "cpu_per_node": 1,
        "gpu_per_node": 0,
        "queue_name": "local",
        "group_size": 1,
        "envs": {},
        "strategy": {"ratio_unfinished": 0.0},
    }

    utils.build_submission(str(cooking_path), machine_info, resrc_info, task_list=[])

    submission = state["submission_cls"].last_instance
    assert submission.run_calls == 1
    assert len(state["machine_cls"].load_calls) == 1

    loaded_machine = state["machine_cls"].load_calls[0]
    assert loaded_machine["local_root"] != loaded_machine["remote_root"]
    assert "local" in Path(loaded_machine["local_root"]).parts
    assert "remote" in Path(loaded_machine["remote_root"]).parts
    assert os.path.isdir(loaded_machine["local_root"])
    assert os.path.isdir(loaded_machine["remote_root"])
    assert "allow_ref" in inspect.signature(state["argument_cls"].normalize_value).parameters
    assert "allow_ref" in inspect.signature(state["argument_cls"].check_value).parameters
