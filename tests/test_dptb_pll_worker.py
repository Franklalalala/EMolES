import importlib
import pickle
import queue
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from ase import Atoms
from ase.db import connect


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_dptb_shims(monkeypatch):
    torch_mod = types.ModuleType("torch")

    @contextmanager
    def _no_grad():
        yield

    torch_mod.no_grad = _no_grad
    torch_mod.device = lambda value: value
    torch_mod.tensor = lambda *args, **kwargs: None
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    dftio_mod = types.ModuleType("dftio")
    dftio_data_mod = types.ModuleType("dftio.data")
    dftio_data_mod._keys = types.SimpleNamespace(
        ATOMIC_NUMBERS_KEY="atomic_numbers",
        PBC_KEY="pbc",
        POSITIONS_KEY="positions",
        CELL_KEY="cell",
    )
    monkeypatch.setitem(sys.modules, "dftio", dftio_mod)
    monkeypatch.setitem(sys.modules, "dftio.data", dftio_data_mod)

    dptb_mod = types.ModuleType("dptb")
    dptb_data_mod = types.ModuleType("dptb.data")
    dptb_data_build_mod = types.ModuleType("dptb.data.build")
    dptb_nn_mod = types.ModuleType("dptb.nn")
    dptb_nn_build_mod = types.ModuleType("dptb.nn.build")
    dptb_nn_hr2hk_mod = types.ModuleType("dptb.nn.hr2hk")

    class _DummyLoader(list):
        def __init__(self, dataset=None, batch_size=None, shuffle=None):
            super().__init__(dataset or [])

    class _DummyProjector:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, batch_info):
            return batch_info

    dptb_data_mod.AtomicData = types.SimpleNamespace(to_AtomicDataDict=lambda value: value)
    dptb_data_mod.AtomicDataDict = types.SimpleNamespace(
        EDGE_FEATURES_KEY="edge_features",
        NODE_FEATURES_KEY="node_features",
        HAMILTONIAN_KEY="hamiltonian",
        EDGE_OVERLAP_KEY="edge_overlap",
        NODE_OVERLAP_KEY="node_overlap",
        OVERLAP_KEY="overlap",
    )
    dptb_data_mod.DataLoader = _DummyLoader
    dptb_data_build_mod.build_dataset = lambda *args, **kwargs: []
    dptb_nn_build_mod.build_model = lambda *args, **kwargs: object()
    dptb_nn_hr2hk_mod.HR2HK = _DummyProjector
    dptb_nn_hr2hk_mod.HR2HK_Gamma_Only = _DummyProjector

    monkeypatch.setitem(sys.modules, "dptb", dptb_mod)
    monkeypatch.setitem(sys.modules, "dptb.data", dptb_data_mod)
    monkeypatch.setitem(sys.modules, "dptb.data.build", dptb_data_build_mod)
    monkeypatch.setitem(sys.modules, "dptb.nn", dptb_nn_mod)
    monkeypatch.setitem(sys.modules, "dptb.nn.build", dptb_nn_build_mod)
    monkeypatch.setitem(sys.modules, "dptb.nn.hr2hk", dptb_nn_hr2hk_mod)


def test_dptb_pll_worker_builds_single_input_lmdb_per_worker(tmp_path, monkeypatch):
    _install_dptb_shims(monkeypatch)
    sys.modules.pop("emoles.inference.model_io", None)
    sys.modules.pop("emoles.inference.dptb_pll", None)

    dptb_pll = importlib.import_module("emoles.inference.dptb_pll")

    db_path = tmp_path / "tiny.db"
    with connect(db_path) as db:
        db.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]), name="a")
        db.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]]), name="b")

    items_path = tmp_path / "worker_00" / "items.json"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_text(
        (
            '{\n'
            '  "items": [\n'
            '    {"source_idx": 0, "source_row_id": 1, "sample_id": 1},\n'
            '    {"source_idx": 1, "source_row_id": 2, "sample_id": 2}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    call_state = {"lmdb_builds": 0, "loader_builds": 0}

    def _fake_prepare_model(checkpoint_path, device):
        return object(), device, "fake-basis", 5.0

    def _fake_build_projectors(model, device, has_overlap):
        return {}

    def _fake_ase_db_to_lmdb(ase_db_path, dptb_lmdb_path, txn_batch_size=128, items=None):
        call_state["lmdb_builds"] += 1
        Path(dptb_lmdb_path).mkdir(parents=True, exist_ok=True)
        (Path(dptb_lmdb_path) / "data.fake.lmdb").write_text("ok", encoding="utf-8")
        return [
            {
                "source_idx": int(item["source_idx"]),
                "source_row_id": int(item["source_row_id"]),
                "sample_id": int(item["sample_id"]),
            }
            for item in items
        ]

    def _fake_prepare_loader(lmdb_path, basis, r_max):
        call_state["loader_builds"] += 1
        return [0, 1]

    def _fake_iter_batches(reference_loader, model, device, max_items):
        for idx in reference_loader:
            yield idx, {"predicted": idx}

    def _fake_save_info(txn, idx, source_metadata, batch_info, model, device, has_overlap=False, projectors=None):
        txn.put(
            str(source_metadata["source_row_id"]).encode("utf-8"),
            pickle.dumps({"source_row_id": int(source_metadata["source_row_id"])}),
        )

    env_store = {}

    class _FakeTxn:
        def __init__(self, store):
            self.store = store

        def put(self, key, value):
            self.store[key] = value
            return True

        def commit(self):
            return None

        def abort(self):
            return None

        def stat(self):
            return {"entries": len(self.store)}

    class _FakeEnv:
        def __init__(self, path):
            self.path = path
            env_store.setdefault(path, {})

        def begin(self, write=False):
            return _FakeTxn(env_store[self.path])

        def close(self):
            return None

    monkeypatch.setattr(dptb_pll, "_prepare_dptb_model", _fake_prepare_model)
    monkeypatch.setattr(dptb_pll, "_build_gamma_projectors", _fake_build_projectors)
    monkeypatch.setattr(dptb_pll, "ase_db_2_dummy_dptb_lmdb", _fake_ase_db_to_lmdb)
    monkeypatch.setattr(dptb_pll, "_prepare_reference_loader", _fake_prepare_loader)
    monkeypatch.setattr(dptb_pll, "_iter_predicted_batches", _fake_iter_batches)
    monkeypatch.setattr(dptb_pll, "save_info_2_lmdb", _fake_save_info)
    monkeypatch.setattr(dptb_pll, "configure_worker_env", lambda gpu_id=None, cpu_threads_per_worker=None: None)
    monkeypatch.setattr(
        dptb_pll,
        "reset_lmdb_directory",
        lambda path: Path(path).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        dptb_pll,
        "open_lmdb_environment",
        lambda path, readonly=False: _FakeEnv(path),
    )

    worker_spec = {
        "worker_id": 0,
        "worker_name": "worker_00",
        "worker_root": str(items_path.parent),
        "items_path": str(items_path),
        "num_items": 2,
        "gpu_id": None,
        "ase_db_path": str(db_path),
        "checkpoint_path": str(tmp_path / "fake_ckpt.pth"),
        "infer_root": str(tmp_path / "infer"),
        "infer_lmdb_path": str(tmp_path / "infer" / "worker_00.lmdb"),
        "input_lmdb_root": str(tmp_path / "infer_input_lmdb" / "worker_00"),
        "has_overlap": False,
        "txn_batch_size": 16,
        "input_txn_batch_size": 16,
        "cleanup_input_lmdb": False,
    }

    result_queue = queue.Queue()
    dptb_pll._run_dptb_slot_worker(
        slot_id=0,
        attempt=0,
        worker_spec=worker_spec,
        cpu_threads_per_worker=1,
        result_queue=result_queue,
    )

    messages = []
    while not result_queue.empty():
        messages.append(result_queue.get())

    message_types = [msg["type"] for msg in messages]
    assert "worker_ready" in message_types
    assert "worker_done" in message_types
    assert "worker_fatal" not in message_types
    assert call_state["lmdb_builds"] == 1
    assert call_state["loader_builds"] == 1
    assert len(env_store[worker_spec["infer_lmdb_path"]]) == 2
    assert (Path(worker_spec["input_lmdb_root"]) / "data.fake.lmdb").exists()
