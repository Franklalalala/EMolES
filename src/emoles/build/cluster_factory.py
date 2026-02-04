import os
import re
import argparse
import random
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import write
from tqdm import tqdm

# ==========================================
# Imports & Dependency Handling
# ==========================================

# 1. Core Logic Import
from emoles.build.cluster import build_cluster

# 2. UMA Import (Optional/Safe)
try:
    import emoles.build.uma_entry as uma_entry

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    uma_entry = None

# 3. RDKit for Fallback (Required by emoles anyway)
from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# Default Constants
# ==========================================
DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"


# ==========================================
# Helper Functions
# ==========================================

def sanitize_filename(filename: str, max_length: int = 30) -> str:
    """Shorten names for complex filenames."""
    sanitized = re.sub(r'[^\w\-.]', '', filename)
    if len(sanitized) > max_length:
        return sanitized[:max_length]
    return sanitized or "mol"


def _fallback_smiles_to_atoms(smiles: str) -> Atoms:
    """
    Lightweight fallback to convert SMILES to ASE Atoms without UMA.
    Used when --no-uma is set or UMA is missing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    # Basic embedding
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        pass

    # Convert to ASE
    conf = mol.GetConformer()
    positions = []
    symbols = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())

    return Atoms(symbols=symbols, positions=positions)


def load_db_entries(db_path: str, show_progress: bool = True) -> List[Dict]:
    """Load entries from an ASE database file."""
    entries = []
    if db_path is None or not os.path.exists(db_path):
        return entries

    with connect(db_path) as db:
        total_rows = db.count()
        rows = db.select()
        if show_progress and total_rows > 0:
            rows = tqdm(rows, total=total_rows, desc=f"Loading {os.path.basename(db_path)}", unit="entry")

        for row in rows:
            name = row.get('name', f"row_{row.id}")
            entries.append({
                'id': row.id,
                'name': name,
                'atoms': row.toatoms(),
            })
    return entries


def optimize_monomers(
        entries: List[Dict],
        prefix: str,
        root_workspace: str,
        device: str,
        use_uma: bool
) -> List[Dict]:
    """
    Prepare monomers.
    If use_uma is True and Available -> Run UMA optimization.
    Else -> Convert SMILES to Atoms using lightweight RDKit fallback.
    """

    # --- Branch 1: NO UMA (Fast Path / Fallback) ---
    if not use_uma or not UMA_AVAILABLE:
        if use_uma and not UMA_AVAILABLE:
            print(f"[Warning] UMA requested but not installed. Falling back to basic RDKit embedding for {prefix}.")
        else:
            print(f"[Info] UMA skipped for {prefix}. Using basic RDKit embedding.")

        processed_entries = []
        for ent in entries:
            atoms_obj = ent['atoms']
            # If input is string, convert it
            if isinstance(atoms_obj, str):
                try:
                    atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except Exception as e:
                    print(f"  Error converting {ent['name']}: {e}")
                    continue

            # Tag info
            atoms_obj.info['n_anion'] = 1 if prefix.lower() == "anion" else 0

            processed_entries.append({
                'id': ent['id'],
                'name': ent['name'],
                'atoms': atoms_obj
            })
        return processed_entries

    # --- Branch 2: USE UMA (Optimization Path) ---
    print(f"\n[Pre-Optimization] detected SMILES input for {prefix}. Optimizing monomers with UMA...")
    temp_workspace = os.path.join(root_workspace, f"temp_opt_{prefix.lower()}")
    os.makedirs(temp_workspace, exist_ok=True)
    input_db_path = os.path.join(temp_workspace, "raw_monomers.db")

    if os.path.exists(input_db_path): os.remove(input_db_path)

    with connect(input_db_path) as db:
        for ent in entries:
            atoms_obj = ent['atoms']
            if isinstance(atoms_obj, str):
                try:
                    # Use UMA's internal converter if available, or fallback
                    if hasattr(uma_entry, 'smiles_to_atoms'):
                        atoms_obj = uma_entry.smiles_to_atoms(atoms_obj)
                    else:
                        atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except Exception as e:
                    print(f"  Error embedding {ent['name']}: {e}")
                    continue

            atoms_obj.info['n_anion'] = 1 if prefix.lower() == "anion" else 0
            db.write(atoms_obj, name=ent['name'])

    optimized_db_path = uma_entry.entry(
        input_db=input_db_path,
        workspace=temp_workspace,
        device=device,
        verbose=False,
        show_progress=True
    )

    if optimized_db_path is None:
        optimized_db_path = os.path.join(temp_workspace, "optimized_all.db")

    if not os.path.exists(optimized_db_path):
        print(f"[Warning] Optimized DB not found for {prefix}. Using input structures.")
        # Re-read raw if opt failed
        return load_db_entries(input_db_path, show_progress=False)

    return load_db_entries(optimized_db_path, show_progress=False)


def parse_smiles_input(smiles_list: List[str], default_prefix: str) -> List[Dict]:
    entries = []
    if not smiles_list: return entries
    for i, item in enumerate(smiles_list):
        if ':' in item:
            smiles, name = item.split(':', 1)
        else:
            smiles, name = item, f"{default_prefix}_{i + 1}"
        entries.append({'id': i, 'name': name.strip(), 'atoms': smiles.strip()})
    return entries


def normalize_input_data(source, prefix, show_progress, workspace, device, use_uma) -> List[Dict]:
    if source is None: return []
    if isinstance(source, list) and len(source) > 0 and isinstance(source[0], dict): return source

    if isinstance(source, str) and (source.endswith('.db') or source.endswith('.json')):
        print(f"Loading {prefix} from DB: {source}")
        return load_db_entries(source, show_progress)

    s_list = [source] if isinstance(source, str) else source
    raw_entries = parse_smiles_input(s_list, prefix)

    # Pass the UMA flag down
    return optimize_monomers(raw_entries, prefix, workspace, device, use_uma)


# ==========================================
# Combinatorial Logic
# ==========================================

def integer_partitions(target: int, k: int, min_val: int = 1) -> Generator[Tuple[int, ...], None, None]:
    """Generate all ways to sum to 'target' using 'k' integers, each >= min_val."""
    if k == 1:
        if target >= min_val:
            yield (target,)
        return

    upper_bound = target - (k - 1) * min_val
    for i in range(min_val, upper_bound + 1):
        for tail in integer_partitions(target - i, k - 1, min_val):
            yield (i,) + tail


def plan_mixtures(
        solvents_pool: List[Dict],
        anions_pool: List[Dict],
        mix_n: int,
        num_mixtures: int,
        target_totals: Tuple[int, ...],
        anion_counts: Tuple[int, ...],
        seed: int = 42
) -> List[Dict]:
    """Generates a build plan for a SPECIFIC mix_n."""
    plan = []
    random.seed(seed)

    if mix_n > len(solvents_pool):
        return []

    import math
    n_total_combos = math.comb(len(solvents_pool), mix_n)

    solvent_combinations = []
    if mix_n == 1:
        solvent_combinations = [(s,) for s in solvents_pool]
    elif n_total_combos <= num_mixtures * 2:
        solvent_combinations = list(itertools.combinations(solvents_pool, mix_n))
    else:
        seen = set()
        attempts = 0
        while len(solvent_combinations) < num_mixtures and attempts < num_mixtures * 10:
            combo = tuple(sorted(random.sample(solvents_pool, mix_n), key=lambda x: x['name']))
            if combo not in seen:
                seen.add(combo)
                solvent_combinations.append(combo)
            attempts += 1

    for solvents_tuple in solvent_combinations:
        current_anions_loop = anions_pool if anions_pool else [None]

        for anion_entry in current_anions_loop:
            for total_coord in target_totals:
                valid_anion_counts = [ac for ac in anion_counts if ac <= total_coord]
                if not anion_entry: valid_anion_counts = [0]

                for n_anion in valid_anion_counts:
                    n_solvent_total = total_coord - n_anion

                    if n_solvent_total < mix_n:
                        continue

                    partitions = list(integer_partitions(n_solvent_total, mix_n, min_val=1))

                    for p in partitions:
                        ligands_def = []
                        for idx, s_ent in enumerate(solvents_tuple):
                            ligands_def.append({
                                'type': 'solvent',
                                'entry': s_ent,
                                'count': p[idx]
                            })

                        if anion_entry and n_anion > 0:
                            ligands_def.append({
                                'type': 'anion',
                                'entry': anion_entry,
                                'count': n_anion
                            })

                        charge = 1 - n_anion
                        if n_anion == 0:
                            cat = "SSIP"
                        elif n_anion == 1:
                            cat = "CIP"
                        else:
                            cat = "AGG"

                        plan.append({
                            'category': cat,
                            'ligands': ligands_def,
                            'total_coord': total_coord,
                            'n_anion_total': n_anion,
                            'charge': charge,
                            'mix_type': f"Mix-{mix_n}"
                        })
    return plan


def compose_filename(ion: str, plan_item: Dict) -> str:
    parts = [ion, plan_item['category']]

    solv_idx = 1
    for lig in plan_item['ligands']:
        if lig['type'] == 'solvent':
            name = sanitize_filename(lig['entry']['name'])
            parts.append(f"S{solv_idx}-{name}_n{solv_idx}-{lig['count']}")
            solv_idx += 1

    for lig in plan_item['ligands']:
        if lig['type'] == 'anion':
            name = sanitize_filename(lig['entry']['name'])
            parts.append(f"A-{name}_na-{lig['count']}")

    return "_".join(parts) + ".xyz"


def build_from_plan(
        plan: List[Dict],
        out_dir: Path,
        ion: str,
        cluster_kwargs: Dict,
        show_progress: bool,
) -> Dict[str, int]:
    stats = {'attempted': 0, 'built': 0, 'failed': 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "structures.db"
    if db_path.exists(): os.remove(db_path)
    db = connect(db_path)

    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(exist_ok=True)

    iter_obj = tqdm(plan, desc="Building Clusters", unit="item") if show_progress else plan

    for item in iter_obj:
        stats['attempted'] += 1

        ligand_info_arg = []
        for lig in item['ligands']:
            atoms_obj = lig['entry']['atoms']
            # Make a copy just in case
            if isinstance(atoms_obj, Atoms):
                atoms_obj = atoms_obj.copy()

            if lig['type'] == 'anion' and isinstance(atoms_obj, Atoms):
                atoms_obj.charge = -1
                atoms_obj.set_initial_charges(np.full(len(atoms_obj), -1 / len(atoms_obj)))
            ligand_info_arg.append((atoms_obj, lig['count']))

        fname = compose_filename(ion, item)
        if show_progress:
            iter_obj.set_postfix_str(f"{fname[:30]}...")

        try:
            cluster = build_cluster(
                ion_identifier=ion,
                ligand_molecule_info=ligand_info_arg,
                **cluster_kwargs
            )

            cluster.info['charge'] = item['charge']
            cluster.info['category'] = item['category']

            ion_symbol = ''.join([c for c in ion if c.isalpha()])
            for atom in cluster:
                if atom.symbol == ion_symbol:
                    atom.charge = 1.0
                    break

            write(str(xyz_dir / fname), cluster)

            kvp = {
                'category': item['category'],
                'ion': ion,
                'charge': item['charge'],
                'total_coord': item['total_coord'],
                'n_atoms': len(cluster),
                'filename': fname,
                'mix_type': item.get('mix_type', 'unknown')
            }
            for i, lig in enumerate(item['ligands']):
                kvp[f"lig_{i}_name"] = lig['entry']['name']
                kvp[f"lig_{i}_type"] = lig['type']
                kvp[f"lig_{i}_count"] = lig['count']

            db.write(cluster, data=kvp, **kvp)
            stats['built'] += 1

        except Exception as e:
            stats['failed'] += 1
            # print(f"Failed {fname}: {e}")

    return stats


def entry(
        solvents: Union[str, List[str]],
        anions: Union[str, List[str]],
        out_dir: str = 'out_mixture',
        ion: str = 'Li',
        target_totals: Tuple[int, ...] = (4, 5),
        anion_counts: Tuple[int, ...] = (1,),
        mix_n_list: Tuple[int, ...] = (1,),
        num_mixtures: int = 10,
        use_uma: bool = True,  # Renamed/Changed logic
        device: str = "cuda",
        verbose: bool = True,
        show_progress: bool = True,
        **cluster_kwargs
):
    # 1. Load Data (Passing use_uma flag)
    solv_data = normalize_input_data(solvents, "Solvent", show_progress, out_dir, device, use_uma)
    anion_data = normalize_input_data(anions, "Anion", show_progress, out_dir, device, use_uma)

    if not solv_data:
        raise ValueError("No solvent data found.")

    # 2. Plan
    full_plan = []
    print(f"\nGenerating Plans for Mix sizes: {mix_n_list}")

    for m_n in mix_n_list:
        sub_plan = plan_mixtures(
            solvents_pool=solv_data,
            anions_pool=anion_data,
            mix_n=m_n,
            num_mixtures=num_mixtures,
            target_totals=target_totals,
            anion_counts=anion_counts
        )
        full_plan.extend(sub_plan)

    print(f"Total Plan: {len(full_plan)} configurations.")
    if len(full_plan) == 0:
        print("Plan is empty. Check constraints.")
        return

    # 3. Build (Strict hyperparameters from snippet)
    final_kwargs = dict(
        relative_score_threshold=0.8,
        max_patch_atoms=2,
        initial_sphere_skin_factor=0.7,
        sphere_skin_increment_factor=0.01,
        target_no_clashes=True,
        rotation_opt_iterations=50,
        verbose=False
    )
    final_kwargs.update(cluster_kwargs)

    out_path = Path(out_dir)
    stats = build_from_plan(full_plan, out_path, ion, final_kwargs, show_progress)

    print(f"Build Done: {stats['built']}/{stats['attempted']} success.")

    # 4. Post-Build Optimization (UMA) - only if requested and available
    if use_uma and UMA_AVAILABLE and stats['built'] > 0:
        raw_db = str(out_path / "structures.db")
        opt_dir = out_path / "optimized"
        print(f"\nRunning UMA Optimization on {raw_db}...")
        try:
            uma_entry.entry(
                input_db=raw_db,
                workspace=str(opt_dir),
                device=device,
                verbose=verbose,
                show_progress=show_progress
            )
        except Exception as e:
            print(f"UMA Optimization crashed: {e}")
    elif use_uma and not UMA_AVAILABLE:
        print("\n[Warning] Post-build optimization skipped because 'uma_entry' module is missing.")


def main():
    parser = argparse.ArgumentParser(description="Build Multi-Component Clusters")

    # Inputs
    parser.add_argument('--solvents', nargs='*', default=None, help="SMILES or DB path for solvents")
    parser.add_argument('--anions', nargs='*', default=None, help="SMILES or DB path for anions")
    parser.add_argument('--out', default='out_mixture', help="Output directory")

    # Configuration
    parser.add_argument('--ion', default='Li', help="Ion identifier")
    parser.add_argument('--target-totals', default='4,5', help="Allowed total coordination numbers")
    parser.add_argument('--anion-counts', default='1', help="Allowed anion counts")

    # Mixture Logic
    parser.add_argument('--mix-n', default='1', help="List of mix sizes (e.g. '1,2')")
    parser.add_argument('--num-mixtures', type=int, default=10, help="Max random solvent combinations")

    # Flags - CHANGED: --no-opt to --no-uma
    parser.add_argument('--no-uma', action='store_false', dest='use_uma', help="Disable UMA pre/post-optimization")
    # Default is use_uma=True unless --no-uma is passed
    parser.set_defaults(use_uma=True)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Parsing lists
    solv_arg = args.solvents if args.solvents else [DEFAULT_DME_SMILES]
    if len(solv_arg) == 1 and (solv_arg[0].endswith('.db') or solv_arg[0].endswith('.json')):
        solv_arg = solv_arg[0]

    anion_arg = args.anions if args.anions else [DEFAULT_FSI_SMILES]
    if len(anion_arg) == 1 and (anion_arg[0].endswith('.db') or anion_arg[0].endswith('.json')):
        anion_arg = anion_arg[0]

    t_totals = tuple(int(x) for x in args.target_totals.split(',') if x.strip())
    a_counts = tuple(int(x) for x in args.anion_counts.split(',') if x.strip())
    m_n_list = tuple(int(x) for x in args.mix_n.split(',') if x.strip())

    entry(
        solvents=solv_arg,
        anions=anion_arg,
        out_dir=args.out,
        ion=args.ion,
        target_totals=t_totals,
        anion_counts=a_counts,
        mix_n_list=m_n_list,
        num_mixtures=args.num_mixtures,
        use_uma=args.use_uma,  # New flag
        device=args.device,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()