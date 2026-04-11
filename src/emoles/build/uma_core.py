import os
import re
import shutil
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.db.core import check as ase_db_check

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies
from fairchem.core.units.mlip_unit import load_predict_unit


DEFAULT_CHECKPOINT = r"/home/user/openequi_workspace/my_run/checkpoints/uma-m-1p1.pt"
DEFAULT_WORKSPACE = os.path.abspath("out_li_clusters")
DEFAULT_MODEL_NAME = "uma-m-1p1"


def sanitize_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unnamed"


def _coerce_int(x, default=None):
    if x is None:
        return default
    try:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return int(float(x))
        return int(float(x))
    except Exception:
        return default


def _normalize_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _is_db_scalar(v) -> bool:
    v = _normalize_scalar(v)
    return isinstance(v, (str, int, float, bool))


def _jsonify(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return repr(obj)


def _prepare_db_key_value_pairs(meta: dict) -> dict:
    kvp = {}
    for key, value in (meta or {}).items():
        if not isinstance(key, str):
            continue
        value = _normalize_scalar(value)
        if value is None or (not _is_db_scalar(value)):
            continue
        try:
            ase_db_check({key: value})
            kvp[key] = value
        except Exception:
            continue
    return kvp


def _safe_log(msg: str, show_progress: bool = True):
    from tqdm import tqdm

    if show_progress:
        tqdm.write(msg)
    else:
        print(msg)


def _derive_names(row, meta: dict):
    raw_name = meta.get("xyz_file", None) or meta.get("name", None) or f"id_{row.id}"
    raw_name = os.path.splitext(str(raw_name))[0]
    base_name = sanitize_name(raw_name)
    unique_name = f"id_{int(row.id):06d}__{base_name}"
    return base_name, unique_name


def smiles_to_atoms(smiles: str) -> Atoms:
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to process SMILES strings.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res == -1:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if res == -1:
        raise RuntimeError(f"RDKit failed to embed SMILES: {smiles}")
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def prepare_input_source(workspace: str, input_db: str = None, smiles_list: list = None) -> str:
    os.makedirs(workspace, exist_ok=True)
    if smiles_list:
        temp_db_path = os.path.join(workspace, "temp_smiles_input.db")
        print(f"Generating 3D structures from {len(smiles_list)} SMILES -> {temp_db_path} ...")
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        with connect(temp_db_path) as db:
            for idx, smi in enumerate(smiles_list):
                try:
                    atoms = smiles_to_atoms(smi)
                    name = f"smiles_{idx}_{sanitize_name(smi[:10])}"
                    db.write(atoms, name=name, smiles=smi)
                except Exception as exc:
                    print(f"[Warning] Failed to convert SMILES '{smi}': {exc}")
        return temp_db_path
    if not input_db:
        return os.path.join(workspace, "all.db")
    return input_db


def _get_row_key_value_pairs(row) -> dict:
    try:
        kvp = getattr(row, "key_value_pairs", None)
        return dict(kvp) if kvp else {}
    except Exception:
        return {}


def _get_row_data(row) -> dict:
    try:
        data = getattr(row, "data", None)
        return dict(data) if data else {}
    except Exception:
        return {}


def _merge_row_metadata(row, atoms: Atoms) -> dict:
    meta = {}
    meta.update(_get_row_key_value_pairs(row))
    meta.update(_get_row_data(row))
    try:
        if isinstance(getattr(atoms, "info", None), dict):
            for key, value in atoms.info.items():
                if key not in meta:
                    meta[key] = value
    except Exception:
        pass
    return meta


def _infer_charge(atoms: Atoms, meta: dict) -> int:
    if "charge" in meta and meta["charge"] is not None:
        charge = _coerce_int(meta["charge"], default=None)
        if charge is not None:
            return int(charge)

    n_anion = None
    for key in ("n_anion", "n_anion_total", "n_anions", "n_anion_tot"):
        if key in meta and meta[key] is not None:
            n_anion = _coerce_int(meta[key], default=None)
            if n_anion is not None:
                break

    has_li = "Li" in atoms.get_chemical_symbols()
    if n_anion is not None:
        return int((1 - n_anion) if has_li else (0 - n_anion))
    return int(1 if has_li else 0)


def _extract_source_row_id_from_row(row) -> int:
    kvp = _get_row_key_value_pairs(row)
    if "source_row_id" in kvp:
        value = _coerce_int(kvp["source_row_id"], None)
        if value is not None:
            return int(value)
    data = _get_row_data(row)
    if "source_row_id" in data:
        value = _coerce_int(data["source_row_id"], None)
        if value is not None:
            return int(value)
    raise KeyError("source_row_id not found in shard row")


def prepare_serial_workspace(workspace: str):
    os.makedirs(workspace, exist_ok=True)
    traj_dir = os.path.join(workspace, "traj")
    out_xyz_dir = os.path.join(workspace, "optimized_xyz_all")
    out_db_path = os.path.join(workspace, "optimized_all.db")

    if os.path.exists(out_db_path):
        os.remove(out_db_path)

    for directory in [traj_dir, out_xyz_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    return traj_dir, out_xyz_dir, out_db_path


def load_fairchem_calculator(checkpoint_path: str, model_name: str, workspace: str, device: str):
    atom_refs = get_isolated_atomic_energies(model_name, workspace)
    predictor = load_predict_unit(checkpoint_path, "default", None, device, atom_refs)
    return FAIRChemCalculator(predictor, task_name="omol")
