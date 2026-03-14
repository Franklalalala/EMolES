import os
import re
import argparse
import random
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io import write
from tqdm import tqdm

from emoles.build.cluster import build_cluster

try:
    import emoles.build.uma_entry as uma_entry

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    uma_entry = None

from rdkit import Chem
from rdkit.Chem import AllChem

DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"


def sanitize_filename(filename: str, max_length: int = 30) -> str:
    sanitized = re.sub(r'[^\w\-.]', '', filename)
    if len(sanitized) > max_length:
        return sanitized[:max_length]
    return sanitized or "mol"


def _fallback_smiles_to_atoms(smiles: str) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res == -1: AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        pass

    conf = mol.GetConformer()
    positions = []
    symbols = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())
    return Atoms(symbols=symbols, positions=positions)


def load_db_entries(db_path: str, show_progress: bool = True) -> List[Dict]:
    entries = []
    if not db_path or not os.path.exists(db_path): return entries
    with connect(db_path) as db:
        total_rows = db.count()
        rows = db.select()
        if show_progress and total_rows > 0:
            rows = tqdm(rows, total=total_rows, desc=f"Loading {os.path.basename(db_path)}", unit="entry")
        for row in rows:
            name = row.get('name', f"row_{row.id}")
            entries.append({'id': row.id, 'name': name, 'atoms': row.toatoms()})
    return entries


def optimize_monomers(entries: List[Dict], prefix: str, root_workspace: str, device: str, use_uma: bool) -> List[Dict]:
    if not use_uma or not UMA_AVAILABLE:
        processed_entries = []
        for ent in entries:
            atoms_obj = ent['atoms']
            if isinstance(atoms_obj, str):
                try:
                    atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except:
                    continue
            atoms_obj.info['n_anion'] = 1 if prefix.lower() == "anion" else 0
            processed_entries.append({'id': ent['id'], 'name': ent['name'], 'atoms': atoms_obj})
        return processed_entries

    temp_workspace = os.path.join(root_workspace, f"temp_opt_{prefix.lower()}")
    os.makedirs(temp_workspace, exist_ok=True)
    input_db_path = os.path.join(temp_workspace, "raw_monomers.db")
    if os.path.exists(input_db_path): os.remove(input_db_path)

    with connect(input_db_path) as db:
        for ent in entries:
            atoms_obj = ent['atoms']
            if isinstance(atoms_obj, str):
                try:
                    if hasattr(uma_entry, 'smiles_to_atoms'):
                        atoms_obj = uma_entry.smiles_to_atoms(atoms_obj)
                    else:
                        atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except:
                    continue
            atoms_obj.info['n_anion'] = 1 if prefix.lower() == "anion" else 0
            db.write(atoms_obj, name=ent['name'])

    optimized_db_path = uma_entry.entry(input_db=input_db_path, workspace=temp_workspace, device=device, verbose=False,
                                        show_progress=True)
    if not optimized_db_path or not os.path.exists(optimized_db_path):
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
        return load_db_entries(source, show_progress)
    s_list = [source] if isinstance(source, str) else source
    raw_entries = parse_smiles_input(s_list, prefix)
    return optimize_monomers(raw_entries, prefix, workspace, device, use_uma)


def integer_partitions(target: int, k: int, min_val: int = 1) -> Generator[Tuple[int, ...], None, None]:
    if k == 1:
        if target >= min_val: yield (target,)
        return
    upper_bound = target - (k - 1) * min_val
    for i in range(min_val, upper_bound + 1):
        for tail in integer_partitions(target - i, k - 1, min_val):
            yield (i,) + tail


def plan_mixtures(
        solvents_pool: List[Dict],
        anions_pool: List[Dict],
        solv_mix_n_list: Tuple[int, ...],
        anion_mix_n_list: Tuple[int, ...],
        num_mixtures: int,
        states: List[Tuple[int, int]],  # [(n_solvent, n_anion)]
        repeats: int = 1,
        seed: int = 42
) -> List[Dict]:
    plan = []
    random.seed(seed)
    unique_signatures = set()

    for s_mix_n in solv_mix_n_list:
        if s_mix_n > len(solvents_pool): continue
        import math
        n_total_combos = math.comb(len(solvents_pool), s_mix_n)
        solvent_combinations = []
        if s_mix_n == 1:
            solvent_combinations = [(s,) for s in solvents_pool]
        elif n_total_combos <= num_mixtures * 2:
            solvent_combinations = list(itertools.combinations(solvents_pool, s_mix_n))
        else:
            seen = set()
            attempts = 0
            while len(solvent_combinations) < num_mixtures and attempts < num_mixtures * 10:
                combo = tuple(sorted(random.sample(solvents_pool, s_mix_n), key=lambda x: x['name']))
                if combo not in seen:
                    seen.add(combo)
                    solvent_combinations.append(combo)
                attempts += 1

        for a_mix_n in anion_mix_n_list:
            anion_combinations = []
            if anions_pool and a_mix_n <= len(anions_pool):
                anion_combinations = list(itertools.combinations(anions_pool, a_mix_n))

            for solvents_tuple in solvent_combinations:
                for n_solvent, n_anion in states:
                    total_coord = n_solvent + n_anion
                    if total_coord == 0: continue

                    if n_solvent > 0 and n_solvent < s_mix_n: continue
                    if n_anion > 0 and n_anion < a_mix_n: continue

                    solv_parts = list(integer_partitions(n_solvent, s_mix_n, 1)) if n_solvent > 0 else [()]

                    if n_anion == 0:
                        anion_parts_list = [(None, ())]
                    else:
                        if not anion_combinations: continue
                        anion_parts_list = []
                        for a_tup in anion_combinations:
                            for ap in integer_partitions(n_anion, a_mix_n, 1):
                                anion_parts_list.append((a_tup, ap))

                    for sp in solv_parts:
                        for a_tup, ap in anion_parts_list:
                            ligands_def = []
                            if n_solvent > 0:
                                for idx, s_ent in enumerate(solvents_tuple):
                                    ligands_def.append({'type': 'solvent', 'entry': s_ent, 'count': sp[idx]})
                            if n_anion > 0 and a_tup is not None:
                                for idx, a_ent in enumerate(a_tup):
                                    ligands_def.append({'type': 'anion', 'entry': a_ent, 'count': ap[idx]})

                            sig_parts = sorted(
                                [f"{lig['type']}:{lig['entry']['name']}:{lig['count']}" for lig in ligands_def])
                            signature = "|".join(sig_parts)

                            if signature not in unique_signatures:
                                unique_signatures.add(signature)
                                charge = 1 - n_anion
                                cat = "SSIP" if n_anion == 0 else ("CIP" if n_anion == 1 else "AGG")

                                for rep in range(repeats):
                                    plan.append({
                                        'category': cat,
                                        'n_solvent_total': n_solvent,
                                        'n_anion_total': n_anion,
                                        'ligands': ligands_def,
                                        'total_coord': total_coord,
                                        'charge': charge,
                                        'mix_type': f"S{s_mix_n}-A{a_mix_n}",
                                        'repeat_idx': rep
                                    })
    return plan


def compose_filename(ion: str, plan_item: Dict) -> str:
    parts = [ion, plan_item['category']]
    solv_idx, anion_idx = 1, 1
    for lig in plan_item['ligands']:
        name = sanitize_filename(lig['entry']['name'])
        if lig['type'] == 'solvent':
            parts.append(f"S{solv_idx}-{name}_n{lig['count']}")
            solv_idx += 1
        elif lig['type'] == 'anion':
            parts.append(f"A{anion_idx}-{name}_n{lig['count']}")
            anion_idx += 1
    parts.append(f"run{plan_item.get('repeat_idx', 0)}")
    return "_".join(parts) + ".xyz"


def _worker_build_task(item: Dict, ion: str, xyz_dir: Path, cluster_kwargs: Dict) -> Tuple[
    bool, Optional[Dict], Optional[str]]:
    fname = compose_filename(ion, item)
    try:
        ligand_info_arg = []
        for lig in item['ligands']:
            atoms_obj = lig['entry']['atoms']
            if isinstance(atoms_obj, Atoms): atoms_obj = atoms_obj.copy()
            if lig['type'] == 'anion' and isinstance(atoms_obj, Atoms):
                atoms_obj.charge = -1
                atoms_obj.set_initial_charges(np.full(len(atoms_obj), -1 / len(atoms_obj)))
            ligand_info_arg.append((atoms_obj, lig['count']))

        cluster = build_cluster(ion_identifier=ion, ligand_molecule_info=ligand_info_arg, **cluster_kwargs)
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
            'mix_type': item.get('mix_type', 'unknown'),
            'repeat_idx': item.get('repeat_idx', 0),
            'n_solvent': item['n_solvent_total'],
            'n_anion': item['n_anion_total']
        }
        for i, lig in enumerate(item['ligands']):
            kvp[f"lig_{i}_name"] = lig['entry']['name']
            kvp[f"lig_{i}_type"] = lig['type']
            kvp[f"lig_{i}_count"] = lig['count']
        return True, (cluster, kvp), None
    except Exception as e:
        return False, None, str(e)


def build_from_plan(plan: List[Dict], out_dir: Path, ion: str, cluster_kwargs: Dict, show_progress: bool,
                    n_jobs: int = 32) -> Dict[str, int]:
    stats = {'attempted': 0, 'built': 0, 'failed': 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "structures.db"
    if db_path.exists(): os.remove(db_path)
    db = connect(db_path)
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(exist_ok=True)

    total_items = len(plan)
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker_build_task, item, ion, xyz_dir, cluster_kwargs): item for item in plan}
        iterator = as_completed(futures)
        if show_progress: iterator = tqdm(iterator, total=total_items, desc="Building Clusters", unit="item")
        for future in iterator:
            stats['attempted'] += 1
            success, data, error = future.result()
            if success:
                cluster_obj, kvp = data
                db.write(cluster_obj, data=kvp, **kvp)
                stats['built'] += 1
            else:
                stats['failed'] += 1
    return stats


def entry(
        solvents: Union[str, List[str]],
        anions: Union[str, List[str]],
        out_dir: str = 'out_mixture',
        ion: str = 'Li',
        states: List[Tuple[int, int]] = [(3, 1), (4, 1)],  # (n_solvent, n_anion)
        mix_n_list: Tuple[int, ...] = (1,),
        anion_mix_n_list: Tuple[int, ...] = (1,),
        num_mixtures: int = 10,
        repeats: int = 1,
        use_uma: bool = True,
        device: str = "cuda",
        verbose: bool = True,
        show_progress: bool = True,
        n_jobs: int = 32,
        **cluster_kwargs
):
    solv_data = normalize_input_data(solvents, "Solvent", show_progress, out_dir, device, use_uma)
    anion_data = normalize_input_data(anions, "Anion", show_progress, out_dir, device, use_uma)

    if not solv_data: raise ValueError("No solvent data found.")

    full_plan = plan_mixtures(
        solvents_pool=solv_data, anions_pool=anion_data,
        solv_mix_n_list=mix_n_list, anion_mix_n_list=anion_mix_n_list,
        num_mixtures=num_mixtures, states=states, repeats=repeats
    )

    total_tasks = len(full_plan)
    if total_tasks == 0:
        print("Plan is empty. Check constraints.")
        return

    # --- Print Preview by States ---
    state_map = {}
    for task in full_plan:
        state_key = (task['n_solvent_total'], task['n_anion_total'])
        if state_key not in state_map: state_map[state_key] = []
        state_map[state_key].append(task)

    print("\n" + "=" * 65)
    print("=== Generation Plan Summary ===")
    print(f"Total Tasks in Pool: {total_tasks}")
    print("-" * 65)

    for state_key in sorted(state_map.keys(), key=lambda x: (x[0] + x[1], x[1])):
        n_solv, n_ani = state_key
        cat = "SSIP" if n_ani == 0 else ("CIP" if n_ani == 1 else "AGG")
        count = len(state_map[state_key])
        pct = (count / total_tasks) * 100
        print(f"  State {n_solv:>2} Solv : {n_ani:>2} Anion ({cat:<4}) | {count:>6} tasks | {pct:>5.1f}%")

    print("-" * 65)
    print("--- Preview (Sampled up to 5 per state) ---")

    for state_key in sorted(state_map.keys(), key=lambda x: (x[0] + x[1], x[1])):
        n_solv, n_ani = state_key
        cat = "SSIP" if n_ani == 0 else ("CIP" if n_ani == 1 else "AGG")
        items = state_map[state_key]

        print(f"\n[State: {n_solv} Solv : {n_ani} Anion ({cat})]")
        sample_tasks = random.sample(items, min(5, len(items)))

        for i, task in enumerate(sample_tasks):
            ligand_strs = [f"{l['count']}x {l['entry']['name']} ({l['type']})" for l in task['ligands']]
            mix_str = " + ".join(ligand_strs)
            print(f"  {i + 1}. Coord={task['total_coord']} | {mix_str} | Repeat: #{task['repeat_idx']}")

    print("=" * 65 + "\n")

    final_kwargs = dict(relative_score_threshold=0.8, max_patch_atoms=2, initial_sphere_skin_factor=0.7,
                        sphere_skin_increment_factor=0.01, target_no_clashes=True, rotation_opt_iterations=50,
                        max_sphere_expansions=100, verbose=False)
    final_kwargs.update(cluster_kwargs)

    out_path = Path(out_dir)
    stats = build_from_plan(full_plan, out_path, ion, final_kwargs, show_progress, n_jobs=n_jobs)
    print(f"\nBuild Done: {stats['built']}/{stats['attempted']} success.")

    if use_uma and UMA_AVAILABLE and stats['built'] > 0:
        raw_db = str(out_path / "structures.db")
        opt_dir = out_path / "optimized"
        print(f"\nRunning UMA Optimization on {raw_db}...")
        try:
            uma_entry.entry(input_db=raw_db, workspace=str(opt_dir), device=device, verbose=verbose,
                            show_progress=show_progress)
        except Exception as e:
            print(f"UMA Optimization crashed: {e}")