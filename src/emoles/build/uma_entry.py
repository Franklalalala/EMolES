# uma_entry.py
import os
import re
import shutil
import argparse
import numpy as np
from ase import Atoms
from ase.io import write
from ase.db import connect
from ase.optimize import LBFGS
from tqdm import tqdm

# Optional import for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.calculate.pretrained_mlip import get_isolated_atomic_energies

# ==========================================
# Default Configuration
# ==========================================
DEFAULT_CHECKPOINT = r'/home/mingkang_nt/hetero_atoms_workspace/checkpoints/uma-m-1p1.pt'
DEFAULT_WORKSPACE = os.path.abspath('out_li_clusters')
DEFAULT_MODEL_NAME = "uma-m-1p1"


def sanitize_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "unnamed"


def _coerce_int(x, default=None):
    """Best-effort conversion to int (handles int/float/np scalars/strings)."""
    if x is None:
        return default
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return int(float(x))
        return int(float(x))
    except Exception:
        return default


def smiles_to_atoms(smiles: str) -> Atoms:
    """Converts a SMILES string to an ASE Atoms object using RDKit."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required to process SMILES strings.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res == -1:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)


def prepare_input_source(workspace: str, input_db: str = None, smiles_list: list = None) -> str:
    """
    Determines the input database.
    If SMILES are provided, creates a temporary DB and returns its path.
    Otherwise, returns the existing input_db path.
    """
    os.makedirs(workspace, exist_ok=True)

    # Case 1: SMILES provided -> Generate temp DB
    if smiles_list:
        temp_db_path = os.path.join(workspace, 'temp_smiles_input.db')
        print(f"Generating 3D structures from {len(smiles_list)} SMILES -> {temp_db_path} ...")

        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

        with connect(temp_db_path) as db:
            for i, smi in enumerate(smiles_list):
                try:
                    atoms = smiles_to_atoms(smi)
                    name = f"smiles_{i}_{sanitize_name(smi[:10])}"
                    db.write(atoms, name=name, smiles=smi)
                except Exception as e:
                    print(f"[Warning] Failed to convert SMILES '{smi}': {e}")

        return temp_db_path

    # Case 2: Use existing DB
    if not input_db:
        return os.path.join(workspace, 'all.db')

    return input_db


def _merge_row_metadata(row, atoms: Atoms) -> dict:
    """
    ASE DB rows can store metadata in multiple places:
      - row.key_value_pairs
      - row.data (dict)
      - atoms.info

    Minimal fix based on your DB inspection:
      structures.db 中 data['charge'] 是正确的，但 key_value_pairs['charge'] 是错误的(常为1)。
      因此合并时必须让 row.data 覆盖 row.key_value_pairs。
    """
    meta = {}

    # --- MINIMAL CHANGE: merge order swapped so row.data overrides kvp ---
    # key_value_pairs first
    try:
        if getattr(row, "key_value_pairs", None):
            meta.update(row.key_value_pairs)
    except Exception:
        pass

    # row.data second (override kvp; especially for 'charge')
    try:
        if getattr(row, "data", None):
            meta.update(row.data)
    except Exception:
        pass

    # atoms.info last fallback (only fill missing keys)
    try:
        if isinstance(getattr(atoms, "info", None), dict):
            for k, v in atoms.info.items():
                if k not in meta:
                    meta[k] = v
    except Exception:
        pass

    return meta


def _infer_charge(atoms: Atoms, meta: dict) -> int:
    """
    Charge priority:
      1) meta['charge'] if present  (now correctly prefers row.data['charge'])
      2) infer from anion count keys: n_anion, n_anion_total, n_anions, n_anion_tot

    Inference rule (monomer vs cluster):
      - if structure contains Li: charge = 1 - n_anion   (Li+ cluster convention)
      - else:                    charge = 0 - n_anion   (monomer convention: solvent 0, anion -1)

    Fallback if no n_anion info:
      - contains Li -> +1
      - else -> 0
    """
    if "charge" in meta and meta["charge"] is not None:
        ch = _coerce_int(meta["charge"], default=None)
        if ch is not None:
            return int(ch)

    n_anion = None
    for key in ("n_anion", "n_anion_total", "n_anions", "n_anion_tot"):
        if key in meta and meta[key] is not None:
            n_anion = _coerce_int(meta[key], default=None)
            if n_anion is not None:
                break

    has_li = ("Li" in atoms.get_chemical_symbols())

    if n_anion is not None:
        return int((1 - n_anion) if has_li else (0 - n_anion))

    return int(1 if has_li else 0)


def entry(
        input_db: str = None,
        smiles: list = None,
        workspace: str = DEFAULT_WORKSPACE,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        fmax: float = 0.05,
        max_steps: int = 200,
        verbose: bool = False,
        show_progress: bool = True
) -> str:
    """
    Main optimization routine.
    Returns: Path to the output database.
    """
    os.makedirs(workspace, exist_ok=True)

    # 1) Path Setup & Cleanup
    traj_dir = os.path.join(workspace, 'traj')
    out_xyz_dir = os.path.join(workspace, 'optimized_xyz_all')
    out_db_path = os.path.join(workspace, 'optimized_all.db')

    if os.path.exists(out_db_path):
        if verbose:
            print(f"Removing existing output DB: {out_db_path}")
        os.remove(out_db_path)

    for d in [traj_dir, out_xyz_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 2) Prepare Input Source (DB or SMILES->DB)
    active_input_db = prepare_input_source(workspace, input_db, smiles)

    if verbose:
        print(f"Workdir: {workspace}\nInput: {active_input_db}\nOutput: {out_db_path}")

    # 3) Model Loading
    if verbose:
        print("Loading FAIRChem model...")
    atom_refs = get_isolated_atomic_energies(DEFAULT_MODEL_NAME, workspace)
    predictor = load_predict_unit(checkpoint_path, "default", None, device, atom_refs)
    calc = FAIRChemCalculator(predictor, task_name="omol")

    # 4) Optimization Loop
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db, connect(out_db_path) as tgt_db:
        total = src_db.count()
        rows = src_db.select()
        if show_progress:
            rows = tqdm(rows, total=total, desc="Optimizing", unit="mol")

        def _log(msg: str):
            # Always print (per your requirement), but use tqdm.write to not break the progress bar
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)
        print(active_input_db)
        for row in rows:
            atoms = row.toatoms()
            charge = row.data.get('charge', None)
            if charge is None:
                print('None data_charge detected.')
                meta = _merge_row_metadata(row, atoms)
                charge = _infer_charge(atoms, meta)
            else:
                meta = _merge_row_metadata(row, atoms)
            spin = 1  # spin 永远是 1，不从任何来源读取/改变
            # Attach to atoms/info + meta (FAIRChem requires int-like types)
            atoms.info['charge'] = int(charge)
            atoms.info['spin'] = int(spin)
            meta['charge'] = int(charge)
            meta['spin'] = int(spin)

            # Determine name early (so the log prints a useful identifier)
            raw_name = meta.get('xyz_file', None) or meta.get('name', None) or f"id_{row.id}"
            raw_name = os.path.splitext(str(raw_name))[0]
            base_name = sanitize_name(raw_name)

            # Print charge/spin before optimization (every structure)
            _log(f"[UMA] About to optimize: {base_name} | charge={charge} spin={spin}")

            # ---- Optimization ----
            atoms.calc = calc
            traj_path = os.path.join(traj_dir, f"{base_name}.traj")

            try:
                logfile = '-' if (verbose and not show_progress) else None
                opt = LBFGS(atoms, trajectory=traj_path, logfile=logfile)
                opt.run(fmax=fmax, steps=max_steps)

                out_xyz = os.path.join(out_xyz_dir, f"{base_name}.xyz")
                write(out_xyz, atoms)

                atoms.calc = None  # detach calculator before storing
                tgt_db.write(atoms, data=meta, **meta)

            except Exception as e:
                _log(f"[Error] {base_name}: {e}")

    return out_db_path


def main():
    parser = argparse.ArgumentParser(description="FAIRChem Optimization Script")
    parser.add_argument('--workspace', default=DEFAULT_WORKSPACE, help='Root output directory')

    # Input sources: DB or SMILES
    parser.add_argument('--input-db', default=None, help='Input ASE database')
    parser.add_argument('--smiles', nargs='*', default=None, help='Input SMILES string(s) to optimize')

    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--device', default='cuda', help='Compute device')
    parser.add_argument('--fmax', type=float, default=0.05, help='Force convergence criteria')
    parser.add_argument('--steps', type=int, default=200, dest='max_steps', help='Max optimization steps')

    parser.add_argument('--verbose', action='store_true', help='Show optimization logs')
    parser.add_argument('--no-progress', action='store_false', dest='progress', help='Disable progress bar')
    parser.set_defaults(progress=True)

    args = parser.parse_args()

    out_path = entry(
        input_db=args.input_db,
        smiles=args.smiles,
        workspace=args.workspace,
        checkpoint_path=args.checkpoint,
        device=args.device,
        fmax=args.fmax,
        max_steps=args.max_steps,
        verbose=args.verbose,
        show_progress=args.progress
    )

    print(f"\nOptimization finished. Output DB: {out_path}")


if __name__ == "__main__":
    main()