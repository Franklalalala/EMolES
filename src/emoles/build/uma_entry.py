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
        # Try random coordinates if embedding fails
        AllChem.EmbedMolecule(mol, useRandomCoords=True)

    # Convert RDKit to ASE
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
    # Case 1: SMILES provided -> Generate temp DB
    if smiles_list:
        temp_db_path = os.path.join(workspace, 'temp_smiles_input.db')
        print(f"Generating 3D structures from {len(smiles_list)} SMILES -> {temp_db_path} ...")

        # Clean previous temp file if exists
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

        with connect(temp_db_path) as db:
            for i, smi in enumerate(smiles_list):
                try:
                    atoms = smiles_to_atoms(smi)
                    # Write to DB. Optimization loop will default to charge=1 if n_anion is missing.
                    name = f"smiles_{i}_{sanitize_name(smi[:10])}"
                    db.write(atoms, name=name, smiles=smi)
                except Exception as e:
                    print(f"[Warning] Failed to convert SMILES '{smi}': {e}")

        return temp_db_path

    # Case 2: Use existing DB
    if not input_db:
        return os.path.join(workspace, 'all.db')

    return input_db


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
    # 1. Path Setup & Cleanup
    traj_dir = os.path.join(workspace, 'traj')
    out_xyz_dir = os.path.join(workspace, 'optimized_xyz_all')
    out_db_path = os.path.join(workspace, 'optimized_all.db')

    # Cleanup: Remove old outputs to ensure a fresh run
    # Note: If reusing workspace for multiple steps, be careful.
    # Here we assume entry() controls its specific workspace folder.
    if os.path.exists(out_db_path):
        if verbose: print(f"Removing existing output DB: {out_db_path}")
        os.remove(out_db_path)

    for d in [traj_dir, out_xyz_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 2. Prepare Input Source (DB or SMILES->DB)
    active_input_db = prepare_input_source(workspace, input_db, smiles)

    if verbose:
        print(f"Workdir: {workspace}\nInput: {active_input_db}\nOutput: {out_db_path}")

    # 3. Model Loading
    if verbose: print("Loading FAIRChem model...")
    # Ensure atom_refs cache is in the workspace or a temp loc to avoid permission issues
    atom_refs = get_isolated_atomic_energies(DEFAULT_MODEL_NAME, workspace)
    predictor = load_predict_unit(checkpoint_path, "default", None, device, atom_refs)
    calc = FAIRChemCalculator(predictor, task_name="omol")

    # 4. Optimization Loop
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db, connect(out_db_path) as tgt_db:
        total = src_db.count()
        rows = src_db.select()
        if show_progress:
            rows = tqdm(rows, total=total, desc="Optimizing", unit="mol")

        for row in rows:
            atoms = row.toatoms()
            kvp = row.key_value_pairs.copy()

            # --- Core Logic: Charge Calculation & Type Enforcing ---
            try:
                # 1. Try to read charge/spin directly
                if 'charge' in kvp:
                    charge = int(float(kvp['charge']))
                elif 'n_anion' in kvp:
                    # Heuristic for cluster: Charge = 1 (Li) - N_Anions * 1
                    n_anion = int(float(kvp['n_anion']))
                    charge = int(1 - n_anion)
                else:
                    # Default: assume neutral or +1 based on context?
                    # For safety in this specific Lithium context, default to +1 (often solvated Li)
                    # unless it looks like a pure anion.
                    charge = 1

                spin = int(kvp.get('spin', 1))  # Default spin doublet for Li+ systems

            except Exception:
                # Fallback safety
                charge, spin = 1, 1

            # Set properties (FAIRChem requires strict int types)
            atoms.info['charge'] = charge
            atoms.info['spin'] = spin
            kvp['charge'] = charge
            kvp['spin'] = spin

            # --- Optimization ---
            atoms.calc = calc

            # Determine name
            if 'xyz_file' in kvp:
                raw_name = kvp['xyz_file'].strip('.xyz')
            elif 'name' in kvp:
                raw_name = kvp['name']
            else:
                raw_name = f"id_{row.id}"

            base_name = sanitize_name(raw_name)
            traj_path = os.path.join(traj_dir, f"{base_name}.traj")

            try:
                # Logfile control: '-' for stdout, None for silence
                logfile = '-' if (verbose and not show_progress) else None

                if verbose and show_progress:
                    tqdm.write(f"--- Opt: {base_name} (Q={charge}) ---")

                opt = LBFGS(atoms, trajectory=traj_path, logfile=logfile)
                opt.run(fmax=fmax, steps=max_steps)

                # Save Results
                out_xyz = os.path.join(out_xyz_dir, f"{base_name}.xyz")
                write(out_xyz, atoms)

                atoms.calc = None  # Clean up calculator before saving to DB
                tgt_db.write(atoms, data=kvp, **kvp)

            except Exception as e:
                msg = f"[Error] {base_name}: {e}"
                if show_progress:
                    tqdm.write(msg)
                else:
                    print(msg)

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