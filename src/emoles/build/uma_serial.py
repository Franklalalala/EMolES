import argparse
import os

from ase.db import connect
from ase.io import write
from ase.optimize import LBFGS
from tqdm import tqdm

from emoles.build.uma_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_WORKSPACE,
    _infer_charge,
    _merge_row_metadata,
    load_fairchem_calculator,
    prepare_input_source,
    prepare_serial_workspace,
    smiles_to_atoms,
    sanitize_name,
)


def entry(
    input_db: str = None,
    smiles: list = None,
    workspace: str = DEFAULT_WORKSPACE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    fmax: float = 0.05,
    max_steps: int = 200,
    verbose: bool = False,
    show_progress: bool = True,
) -> str:
    traj_dir, out_xyz_dir, out_db_path = prepare_serial_workspace(workspace)
    active_input_db = prepare_input_source(workspace, input_db, smiles)

    if verbose:
        print(f"Workdir: {workspace}\nInput: {active_input_db}\nOutput: {out_db_path}")
        print("Loading FAIRChem model...")

    calc = load_fairchem_calculator(
        checkpoint_path=checkpoint_path,
        model_name=DEFAULT_MODEL_NAME,
        workspace=workspace,
        device=device,
    )

    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db, connect(out_db_path) as tgt_db:
        total = src_db.count()
        rows = src_db.select()
        if show_progress:
            rows = tqdm(rows, total=total, desc="Optimizing", unit="mol")

        def _log(msg: str):
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)

        print(active_input_db)
        for row in rows:
            atoms = row.toatoms()
            charge = row.data.get("charge", None)
            if charge is None:
                print("None data_charge detected.")
                meta = _merge_row_metadata(row, atoms)
                charge = _infer_charge(atoms, meta)
            else:
                meta = _merge_row_metadata(row, atoms)

            spin = 1
            atoms.info["charge"] = int(charge)
            atoms.info["spin"] = int(spin)
            meta["charge"] = int(charge)
            meta["spin"] = int(spin)

            raw_name = meta.get("xyz_file", None) or meta.get("name", None) or f"id_{row.id}"
            raw_name = os.path.splitext(str(raw_name))[0]
            base_name = sanitize_name(raw_name)
            _log(f"[UMA] About to optimize: {base_name} | charge={charge} spin={spin}")

            atoms.calc = calc
            traj_path = os.path.join(traj_dir, f"{base_name}.traj")

            try:
                logfile = "-" if (verbose and not show_progress) else None
                opt = LBFGS(atoms, trajectory=traj_path, logfile=logfile)
                opt.run(fmax=fmax, steps=max_steps)

                out_xyz = os.path.join(out_xyz_dir, f"{base_name}.xyz")
                write(out_xyz, atoms)

                atoms.calc = None
                tgt_db.write(atoms, data=meta, **meta)
            except Exception as exc:
                _log(f"[Error] {base_name}: {exc}")

    return out_db_path


def main():
    parser = argparse.ArgumentParser(description="FAIRChem Optimization Script")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Root output directory")
    parser.add_argument("--input-db", default=None, help="Input ASE database")
    parser.add_argument("--smiles", nargs="*", default=None, help="Input SMILES string(s) to optimize")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--device", default="cuda", help="Compute device")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence criteria")
    parser.add_argument("--steps", type=int, default=200, dest="max_steps", help="Max optimization steps")
    parser.add_argument("--verbose", action="store_true", help="Show optimization logs")
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bar")
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
        show_progress=args.progress,
    )
    print(f"\nOptimization finished. Output DB: {out_path}")


__all__ = ["entry", "main", "smiles_to_atoms"]
