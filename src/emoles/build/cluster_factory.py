import os
import re
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

from ase import Atoms
from ase.db import connect
from ase.io import write
from tqdm import tqdm

# Import the build_cluster function
from emoles.build.cluster import build_cluster

# ==========================================
# Default Constants
# ==========================================
DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"


# ==========================================
# Helper Functions
# ==========================================

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize string to be safe for filenames."""
    sanitized = re.sub(r'[^\w\-.]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_.')
    if len(sanitized) > max_length:
        name_parts = sanitized.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            max_name_length = max_length - len(ext) - 1
            sanitized = f"{name[:max_name_length]}.{ext}"
        else:
            sanitized = sanitized[:max_length]
    return sanitized or "unnamed_molecule"


def load_db_entries(db_path: str, show_progress: bool = True) -> List[Dict]:
    """Load entries from an ASE database file."""
    entries = []
    if not os.path.exists(db_path):
        return entries

    with connect(db_path) as db:
        total_rows = db.count()
        db_name = os.path.basename(db_path)
        rows = db.select()
        if show_progress and total_rows > 0:
            rows = tqdm(rows, total=total_rows, desc=f"Loading {db_name}", unit="entry")

        for row in rows:
            name = row.get('name', f"row_{row.id}")
            entries.append({
                'id': row.id,
                'name': name,
                'atoms': row.toatoms(),
                'kvp': row.key_value_pairs.copy() if row.key_value_pairs else {},
                'data': row.data.copy() if row.data else {}
            })
    return entries


def parse_smiles_input(smiles_list: List[str], default_prefix: str) -> List[Dict]:
    """Parse a list of SMILES strings (format: 'SMILES' or 'SMILES:Name')."""
    entries = []
    if not smiles_list:
        return entries

    for i, item in enumerate(smiles_list):
        if ':' in item:
            smiles, name = item.split(':', 1)
            name = name.strip()
            smiles = smiles.strip()
        else:
            smiles = item.strip()
            name = f"{default_prefix}_{i + 1}"

        entries.append({
            'id': i,
            'name': name,
            'atoms': smiles,
            'kvp': {},
            'data': {}
        })
    return entries


def normalize_input_data(source: Union[str, List[str], List[Dict], None],
                         prefix: str,
                         show_progress: bool) -> List[Dict]:
    """Normalize input data."""
    if source is None:
        return []

    if isinstance(source, list) and len(source) > 0 and isinstance(source[0], dict) and 'atoms' in source[0]:
        return source

    if isinstance(source, str):
        if source.endswith('.db') or source.endswith('.json'):
            print(f"Loading {prefix} from DB: {source}")
            return load_db_entries(source, show_progress)
        else:
            return parse_smiles_input([source], prefix)

    if isinstance(source, list) and all(isinstance(x, str) for x in source):
        print(f"Parsing {len(source)} {prefix} SMILES inputs")
        return parse_smiles_input(source, prefix)

    return []


def plan_combinations(
        solvents: List[Dict],
        anions: List[Dict],
        max_total: int,
        choices_solv: Tuple[int, ...],
        choices_agg_anion: Tuple[int, ...],
        include_categories: Tuple[str, ...],
) -> Dict[str, List[Dict]]:
    """Generate the build plan."""
    plan = {'SSIP': [], 'CIP': [], 'AGG': []}
    spin = 1

    # SSIP: Charge = +1 (Li+)
    if "SSIP" in include_categories:
        for s in solvents:
            for n_solv in choices_solv:
                if 1 <= n_solv <= max_total:
                    plan['SSIP'].append({
                        'category': 'SSIP',
                        'solvent': s,
                        'anion': None,
                        'n_solv': n_solv,
                        'n_anion': 0,
                        'charge': 1,
                        'spin': spin,
                    })

    # CIP: Charge = 0 (Li+ + Anion-)
    if "CIP" in include_categories and anions:
        for s in solvents:
            for a in anions:
                for n_solv in choices_solv:
                    n_anion = 1
                    if n_solv >= 1 and (n_solv + n_anion) <= max_total:
                        plan['CIP'].append({
                            'category': 'CIP',
                            'solvent': s,
                            'anion': a,
                            'n_solv': n_solv,
                            'n_anion': n_anion,
                            'charge': 0,
                            'spin': spin,
                        })

    # AGG: Charge = 1 - n_anion
    if "AGG" in include_categories and anions:
        for s in solvents:
            for a in anions:
                for n_anion in choices_agg_anion:
                    for n_solv in choices_solv:
                        if n_solv >= 1 and (n_solv + n_anion) <= max_total:
                            plan['AGG'].append({
                                'category': 'AGG',
                                'solvent': s,
                                'anion': a,
                                'n_solv': n_solv,
                                'n_anion': n_anion,
                                'charge': 1 - n_anion,
                                'spin': spin,
                            })
    return plan


def ensure_dirs(root: Path) -> Dict[str, Dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for cat in ['SSIP', 'CIP', 'AGG', 'ALL']:
        xyz_dir = root / f"xyz_{cat.lower()}"
        db_path = root / f"{cat.lower()}.db" if cat != 'ALL' else root / "all.db"
        xyz_dir.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            os.remove(db_path)
        dirs[cat] = {'xyz': xyz_dir, 'db': db_path}
    return dirs


def compose_filename(ion: str, solvent_name: str, anion_name: Optional[str],
                     n_solv: int, n_anion: int, category: str) -> str:
    s = sanitize_filename(solvent_name)
    if n_anion == 0:
        return f"{ion}_{category}_S-{s}_ns-{n_solv}.xyz"
    else:
        a = sanitize_filename(anion_name)
        return f"{ion}_{category}_S-{s}_ns-{n_solv}_A-{a}_na-{n_anion}.xyz"


def build_from_plan(
        plan: Dict[str, List[Dict]],
        out_dirs: Dict[str, Dict[str, Path]],
        ion: str,
        verbose: bool,
        cluster_kwargs: Dict,
        show_progress: bool,
) -> Dict[str, int]:
    stats = {'attempted': 0, 'built': 0, 'failed': 0, 'SSIP': 0, 'CIP': 0, 'AGG': 0}
    db_handles = {cat: connect(out_dirs[cat]['db']) for cat in ['SSIP', 'CIP', 'AGG', 'ALL']}

    total_items = sum(len(plan.get(cat, [])) for cat in ['SSIP', 'CIP', 'AGG'])
    if show_progress:
        main_pbar = tqdm(total=total_items, desc="Building clusters", unit="cluster")

    for cat in ['SSIP', 'CIP', 'AGG']:
        items = plan.get(cat, [])
        if not items: continue

        items_iter = tqdm(items, desc=f"Building {cat}", unit="item", leave=False) if show_progress else items

        for it in items_iter:
            stats['attempted'] += 1

            solvent_name = it['solvent']['name']
            anion_name = it['anion']['name'] if it['anion'] else 'None'

            if show_progress:
                items_iter.set_postfix_str(f"S={solvent_name[:8]} A={anion_name[:8]} ns={it['n_solv']}")

            ligand_info = []
            ligand_info.append((it['solvent']['atoms'], it['n_solv']))

            if it['anion'] and it['n_anion'] > 0:
                anion_obj = it['anion']['atoms']
                if isinstance(anion_obj, Atoms):
                    anion_obj.charge = -1
                ligand_info.append((anion_obj, it['n_anion']))

            current_kwargs = cluster_kwargs.copy()
            if show_progress: current_kwargs['verbose'] = False

            try:
                cluster = build_cluster(
                    ion_identifier=ion,
                    ligand_molecule_info=ligand_info,
                    **current_kwargs
                )

                # ==========================================
                # FIX 1: Explicitly set charges on the Atoms object
                # ==========================================

                # 1. Set global info properties for ASE/XYZ
                cluster.info['charge'] = it['charge']
                cluster.info['spin'] = it['spin']

                # 2. Identify the cation atom and force its charge to +1
                # This fixes the issue where Li comes out neutral (charge 0).
                # We assume the code logic handles Monovalent cations (Li, Na, K) -> +1
                ion_symbol = ''.join([c for c in ion if c.isalpha()])  # Remove charges like "+" from "Li+"
                cation_found = False
                for atom in cluster:
                    if atom.symbol == ion_symbol:
                        # Force the cation to be +1.
                        # This ensures total charge is correct for SSIP (0 + 1 = 1)
                        atom.charge = 1.0
                        cation_found = True
                        break  # Only set the first one (the central ion)

                # ==========================================
                # Save Files
                # ==========================================

                fname = compose_filename(ion, solvent_name, anion_name, it['n_solv'], it['n_anion'], cat)
                # Ensure comment reflects the corrected charge
                comment = f"cat={cat} chg={it['charge']} spin={it['spin']} ion={ion} ns={it['n_solv']} na={it['n_anion']}"

                # Write XYZ
                write(str(out_dirs[cat]['xyz'] / fname), cluster, comment=comment)
                write(str(out_dirs['ALL']['xyz'] / fname), cluster, comment=comment)

                # ==========================================
                # FIX 2: DB Writing Logic
                # ==========================================
                kvp = {
                    'category': cat, 'ion': ion, 'solvent_name': solvent_name, 'anion_name': anion_name,
                    'n_solv': it['n_solv'], 'n_anion': it['n_anion'], 'charge': it['charge'],
                    'spin': it['spin'], 'xyz_file': fname, 'n_atoms_cluster': len(cluster)
                }

                # Use data=kvp to store the dict reliably as JSON blob
                # Use **kvp to try creating searchable columns (best effort)
                db_handles[cat].write(cluster, data=kvp, **kvp)
                db_handles['ALL'].write(cluster, data=kvp, **kvp)

                stats['built'] += 1
                stats[cat] += 1
                if verbose and not show_progress:
                    print(f"  ✓ {cat}: {fname} (Charge: {it['charge']})")

            except Exception as e:
                stats['failed'] += 1
                if verbose:
                    print(f"  ✗ Failed: {solvent_name} + {anion_name}: {e}")
                    # traceback.print_exc()

            if show_progress:
                main_pbar.update(1)

    if show_progress:
        main_pbar.close()

    return stats


def entry(
        solvents: Union[str, List[str], List[Dict]] = [DEFAULT_DME_SMILES],
        anions: Union[str, List[str], List[Dict], None] = [DEFAULT_FSI_SMILES],
        out_dir: str = 'out_li_clusters',
        ion: str = 'Li',
        solv_counts: Tuple[int, ...] = (1, 2, 3, 4),
        agg_anion_counts: Tuple[int, ...] = (2,),
        max_total_ligands: int = 4,
        categories: Tuple[str, ...] = ('SSIP', 'CIP', 'AGG'),
        plan_only: bool = False,
        verbose: bool = True,
        show_progress: bool = True,
        **cluster_kwargs
) -> Dict[str, int]:
    solvents_data = normalize_input_data(solvents, "Solvent", show_progress)
    anions_data = normalize_input_data(anions, "Anion", show_progress)

    if not solvents_data:
        raise ValueError("Solvents data cannot be empty.")

    plan = plan_combinations(
        solvents=solvents_data,
        anions=anions_data,
        max_total=max_total_ligands,
        choices_solv=solv_counts,
        choices_agg_anion=agg_anion_counts,
        include_categories=tuple(c.upper() for c in categories),
    )

    ssip_n = len(plan['SSIP'])
    cip_n = len(plan['CIP'])
    agg_n = len(plan['AGG'])
    print(f"\nPlan: SSIP={ssip_n}, CIP={cip_n}, AGG={agg_n} => Total: {ssip_n + cip_n + agg_n}")

    if plan_only:
        return {}

    out_path = Path(out_dir)
    out_dirs = ensure_dirs(out_path)

    final_cluster_kwargs = dict(
        relative_score_threshold=0.85,
        max_patch_atoms=3,
        initial_sphere_skin_factor=1.25,
        target_no_clashes=True,
        rotation_opt_iterations=50,
        verbose=False
    )
    final_cluster_kwargs.update(cluster_kwargs)

    stats = build_from_plan(
        plan=plan,
        out_dirs=out_dirs,
        ion=ion,
        verbose=verbose,
        cluster_kwargs=final_cluster_kwargs,
        show_progress=show_progress
    )

    print(f"Done. Success: {stats['built']}/{stats['attempted']}. Output: {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build Li+ clusters (Entry Wrapper).")

    parser.add_argument('--solvents', nargs='*', help='SMILES list or DB path. Default: DME', default=None)
    parser.add_argument('--anions', nargs='*', help='SMILES list or DB path. Default: FSI', default=None)
    parser.add_argument('--out', default='out_li_clusters', help='Output dir')
    parser.add_argument('--ion', default='Li', help='Ion identifier')

    parser.add_argument('--max-total', type=int, default=4, help='Max total ligands')
    parser.add_argument('--solv-counts', default='1,2,3,4', help='Solvent counts')
    parser.add_argument('--agg-anion-counts', default='2', help='AGG anion counts')
    parser.add_argument('--categories', default='SSIP,CIP,AGG', help='Categories to build')

    parser.add_argument('--plan-only', action='store_true')
    # Change: Default verbose is True, using --quiet to silence
    parser.add_argument('--quiet', action='store_false', dest='verbose', help='Disable verbose output')
    parser.add_argument('--no-progress', action='store_true')

    parser.set_defaults(verbose=True)  # Ensure default is True
    args = parser.parse_args()

    solvents_arg = args.solvents
    if solvents_arg and len(solvents_arg) == 1 and (
            solvents_arg[0].endswith('.db') or solvents_arg[0].endswith('.json')):
        solvents_arg = solvents_arg[0]
    elif solvents_arg is None:
        solvents_arg = [DEFAULT_DME_SMILES]

    anions_arg = args.anions
    if anions_arg and len(anions_arg) == 1 and (anions_arg[0].endswith('.db') or anions_arg[0].endswith('.json')):
        anions_arg = anions_arg[0]
    elif anions_arg is None:
        anions_arg = [DEFAULT_FSI_SMILES]

    s_counts = tuple(int(x) for x in args.solv_counts.split(',') if x.strip())
    a_counts = tuple(int(x) for x in args.agg_anion_counts.split(',') if x.strip())
    cats = tuple(x.strip().upper() for x in args.categories.split(',') if x.strip())

    entry(
        solvents=solvents_arg,
        anions=anions_arg,
        out_dir=args.out,
        ion=args.ion,
        solv_counts=s_counts,
        agg_anion_counts=a_counts,
        max_total_ligands=args.max_total,
        categories=cats,
        plan_only=args.plan_only,
        verbose=args.verbose,
        show_progress=not args.no_progress
    )


if __name__ == "__main__":
    main()