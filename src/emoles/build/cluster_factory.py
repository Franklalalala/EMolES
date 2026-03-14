import os
import re
import math
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
from emoles.inference.common_tools import atom_2_smile

try:
    import emoles.build.uma_entry as uma_entry

    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False
    uma_entry = None

from rdkit import Chem
from rdkit.Chem import AllChem

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_DME_SMILES = "COCCOC:DME"
DEFAULT_FSI_SMILES = "F[S](=O)(=O)[N-][S](=O)(=O)F:FSI"
DEFAULT_REF_DB_PATH = "/home/mingkang_nt/hetero_atoms_workspace/dc_0314_build/sol_w_dc.db"
_ANION_REF_DB_PATH = "/home/mingkang_nt/hetero_atoms_workspace/dc_0314_build/anion.db"


# ═══════════════════════════════════════════════════════════════════════════════
# Small utilities
# ═══════════════════════════════════════════════════════════════════════════════
def smile_to_inchi(smile: str) -> str:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smile}")
    return Chem.MolToInchi(mol)


def smile_to_maccs_fp_arr(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(list(fingerprint.ToBitString())).astype(int)


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    a, b = fp1.astype(bool), fp2.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def sanitize_filename(filename: str, max_length: int = 30) -> str:
    sanitized = re.sub(r"[^\w\-.]", "", filename)
    if len(sanitized) > max_length:
        return sanitized[:max_length]
    return sanitized or "mol"


def _fallback_smiles_to_atoms(smiles: str) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    positions, symbols = [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())
    return Atoms(symbols=symbols, positions=positions)


# ═══════════════════════════════════════════════════════════════════════════════
# Reference‑DB InChI lookup (skip UMA when geometry already exists)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_inchi_atoms_map(db_path: str) -> Dict[str, Tuple[str, Atoms]]:
    """Return ``{inchi: (row_name, atoms)}`` built from *db_path*."""
    mapping: Dict[str, Tuple[str, Atoms]] = {}
    if not db_path or not os.path.exists(db_path):
        return mapping

    with connect(db_path) as db:
        for row in db.select():
            kv = dict(row.key_value_pairs) if row.key_value_pairs else {}
            name = kv.get("name") or getattr(row, "name", None) or f"ref_{row.id}"

            inchi = kv.get("inchi") or getattr(row, "inchi", None)
            if not inchi:
                smiles = kv.get("Smiles") or kv.get("smiles")
                if smiles:
                    try:
                        inchi = smile_to_inchi(smiles)
                    except Exception:
                        pass
            if not inchi:
                try:
                    smiles = atom_2_smile(row.toatoms())
                    inchi = smile_to_inchi(smiles)
                except Exception:
                    continue

            if inchi and inchi not in mapping:
                mapping[inchi] = (name, row.toatoms())

    return mapping


def _resolve_entries_via_ref_db(
    entries: List[Dict],
    prefix: str,
    ref_db_path: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Split *entries* into (found_in_ref, still_need_optimization)."""
    if not ref_db_path or not os.path.exists(ref_db_path):
        return [], list(entries)

    inchi_map = _build_inchi_atoms_map(ref_db_path)
    if not inchi_map:
        return [], list(entries)

    is_anion = prefix.lower() == "anion"
    found, remaining = [], []

    for ent in entries:
        atoms_or_smiles = ent["atoms"]
        matched = False

        if isinstance(atoms_or_smiles, str):
            try:
                target_inchi = smile_to_inchi(atoms_or_smiles)
            except Exception:
                target_inchi = None

            if target_inchi and target_inchi in inchi_map:
                ref_name, ref_atoms = inchi_map[target_inchi]
                ref_atoms = ref_atoms.copy()
                ref_atoms.info["n_anion"] = 1 if is_anion else 0
                found.append(
                    {
                        "id": ent["id"],
                        "name": ent["name"],
                        "atoms": ref_atoms,
                        "source": f"ref_db({os.path.basename(ref_db_path)}, matched='{ref_name}')",
                    }
                )
                matched = True

        if not matched:
            remaining.append(ent)

    return found, remaining


# ═══════════════════════════════════════════════════════════════════════════════
# DB loading & SMILES parsing
# ═══════════════════════════════════════════════════════════════════════════════
def load_db_entries(db_path: str, show_progress: bool = True, source_tag: str = "") -> List[Dict]:
    entries = []
    if not db_path or not os.path.exists(db_path):
        return entries

    tag = source_tag or f"input_db({os.path.basename(db_path)})"
    with connect(db_path) as db:
        total_rows = db.count()
        rows = db.select()
        if show_progress and total_rows > 0:
            rows = tqdm(rows, total=total_rows, desc=f"Loading {os.path.basename(db_path)}", unit="entry")
        for row in rows:
            name = row.get("name", None) or getattr(row, "name", None) or f"row_{row.id}"
            entries.append({"id": row.id, "name": name, "atoms": row.toatoms(), "source": tag})
    return entries


def parse_smiles_input(smiles_list: List[str], default_prefix: str) -> List[Dict]:
    entries = []
    if not smiles_list:
        return entries
    for i, item in enumerate(smiles_list):
        if ":" in item:
            smiles, name = item.split(":", 1)
        else:
            smiles, name = item, f"{default_prefix}_{i + 1}"
        entries.append({"id": i, "name": name.strip(), "atoms": smiles.strip()})
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# Monomer optimization (UMA or RDKit fallback)
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_monomers(
    entries: List[Dict],
    prefix: str,
    root_workspace: str,
    device: str,
    use_uma: bool,
) -> List[Dict]:
    is_anion = prefix.lower() == "anion"

    if not use_uma or not UMA_AVAILABLE:
        processed = []
        for ent in entries:
            atoms_obj = ent["atoms"]
            if isinstance(atoms_obj, str):
                try:
                    atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except Exception:
                    continue
            atoms_obj.info["n_anion"] = 1 if is_anion else 0
            processed.append(
                {"id": ent["id"], "name": ent["name"], "atoms": atoms_obj, "source": "fallback_rdkit"}
            )
        return processed

    temp_workspace = os.path.join(root_workspace, f"temp_opt_{prefix.lower()}")
    os.makedirs(temp_workspace, exist_ok=True)
    input_db_path = os.path.join(temp_workspace, "raw_monomers.db")
    if os.path.exists(input_db_path):
        os.remove(input_db_path)

    with connect(input_db_path) as db:
        for ent in entries:
            atoms_obj = ent["atoms"]
            if isinstance(atoms_obj, str):
                try:
                    if hasattr(uma_entry, "smiles_to_atoms"):
                        atoms_obj = uma_entry.smiles_to_atoms(atoms_obj)
                    else:
                        atoms_obj = _fallback_smiles_to_atoms(atoms_obj)
                except Exception:
                    continue
            atoms_obj.info["n_anion"] = 1 if is_anion else 0
            db.write(atoms_obj, name=ent["name"])

    optimized_db_path = uma_entry.entry(
        input_db=input_db_path,
        workspace=temp_workspace,
        device=device,
        verbose=False,
        show_progress=True,
    )

    if not optimized_db_path or not os.path.exists(optimized_db_path):
        return load_db_entries(input_db_path, show_progress=False, source_tag="uma_fallback_raw")
    return load_db_entries(optimized_db_path, show_progress=False, source_tag="uma_optimized")


# ═══════════════════════════════════════════════════════════════════════════════
# Unified input normalisation (ref‑DB lookup → UMA / fallback)
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_input_data(
    source,
    prefix: str,
    show_progress: bool,
    workspace: str,
    device: str,
    use_uma: bool,
    ref_db_for_lookup: str = "",
) -> List[Dict]:
    if source is None:
        return []

    if isinstance(source, list) and source and isinstance(source[0], dict):
        return source

    if isinstance(source, str) and (source.endswith(".db") or source.endswith(".json")):
        return load_db_entries(source, show_progress)

    s_list = [source] if isinstance(source, str) else source
    raw_entries = parse_smiles_input(s_list, prefix)

    found, remaining = _resolve_entries_via_ref_db(raw_entries, prefix, ref_db_for_lookup)

    if remaining:
        optimised = optimize_monomers(remaining, prefix, workspace, device, use_uma)
        found.extend(optimised)

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Mixture planning
# ═══════════════════════════════════════════════════════════════════════════════
def integer_partitions(target: int, k: int, min_val: int = 1) -> Generator[Tuple[int, ...], None, None]:
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
    solvent_mix_sizes: Tuple[int, ...],
    anion_mix_sizes: Tuple[int, ...],
    states: List[Tuple[int, int]],
    repeats: int = 1,
    seed: int = 42,
) -> List[Dict]:
    """
    Enumerate all valid cluster compositions.

    Parameters
    ----------
    solvent_mix_sizes : tuple of int
        How many *distinct* solvent species to mix in one cluster (e.g. (1,) = pure,
        (1, 2) = also try binary mixtures).
    anion_mix_sizes : tuple of int
        How many *distinct* anion species to mix in one cluster.
    states : list of (n_solvent, n_anion)
        Each pair specifies the total molecule count of solvent and anion around the ion.
    """
    plan: List[Dict] = []
    random.seed(seed)
    unique_signatures: set = set()

    for s_mix_n in solvent_mix_sizes:
        if s_mix_n > len(solvents_pool):
            continue

        solvent_combinations = list(itertools.combinations(solvents_pool, s_mix_n))

        for a_mix_n in anion_mix_sizes:
            anion_combinations: list = []
            if anions_pool and a_mix_n <= len(anions_pool):
                anion_combinations = list(itertools.combinations(anions_pool, a_mix_n))

            for solvents_tuple in solvent_combinations:
                for n_solvent, n_anion in states:
                    total_coord = n_solvent + n_anion
                    if total_coord == 0:
                        continue
                    if 0 < n_solvent < s_mix_n:
                        continue
                    if 0 < n_anion < a_mix_n:
                        continue

                    solv_parts = list(integer_partitions(n_solvent, s_mix_n, 1)) if n_solvent > 0 else [()]

                    if n_anion == 0:
                        anion_parts_list = [(None, ())]
                    else:
                        if not anion_combinations:
                            continue
                        anion_parts_list = []
                        for a_tup in anion_combinations:
                            for ap in integer_partitions(n_anion, a_mix_n, 1):
                                anion_parts_list.append((a_tup, ap))

                    for sp in solv_parts:
                        for a_tup, ap in anion_parts_list:
                            ligands_def = []
                            if n_solvent > 0:
                                for idx, s_ent in enumerate(solvents_tuple):
                                    ligands_def.append({"type": "solvent", "entry": s_ent, "count": sp[idx]})
                            if n_anion > 0 and a_tup is not None:
                                for idx, a_ent in enumerate(a_tup):
                                    ligands_def.append({"type": "anion", "entry": a_ent, "count": ap[idx]})

                            sig_parts = sorted(
                                f"{lig['type']}:{lig['entry']['name']}:{lig['count']}" for lig in ligands_def
                            )
                            signature = "|".join(sig_parts)
                            if signature in unique_signatures:
                                continue
                            unique_signatures.add(signature)

                            charge = 1 - n_anion
                            cat = "SSIP" if n_anion == 0 else ("CIP" if n_anion == 1 else "AGG")

                            for rep in range(repeats):
                                plan.append(
                                    {
                                        "category": cat,
                                        "n_solvent_total": n_solvent,
                                        "n_anion_total": n_anion,
                                        "ligands": ligands_def,
                                        "total_coord": total_coord,
                                        "charge": charge,
                                        "mix_type": f"S{s_mix_n}-A{a_mix_n}",
                                        "repeat_idx": rep,
                                    }
                                )
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# Cluster building
# ═══════════════════════════════════════════════════════════════════════════════
def compose_filename(ion: str, plan_item: Dict) -> str:
    parts = [ion, plan_item["category"]]
    solv_idx, anion_idx = 1, 1
    for lig in plan_item["ligands"]:
        name = sanitize_filename(lig["entry"]["name"])
        if lig["type"] == "solvent":
            parts.append(f"S{solv_idx}-{name}_n{lig['count']}")
            solv_idx += 1
        elif lig["type"] == "anion":
            parts.append(f"A{anion_idx}-{name}_n{lig['count']}")
            anion_idx += 1
    parts.append(f"run{plan_item.get('repeat_idx', 0)}")
    return "_".join(parts) + ".xyz"


def _worker_build_task(
    item: Dict,
    ion: str,
    xyz_dir: Path,
    cluster_kwargs: Dict,
) -> Tuple[bool, Optional[Tuple], Optional[str]]:
    fname = compose_filename(ion, item)
    try:
        ligand_info_arg = []
        for lig in item["ligands"]:
            atoms_obj = lig["entry"]["atoms"]
            if isinstance(atoms_obj, Atoms):
                atoms_obj = atoms_obj.copy()
            if lig["type"] == "anion" and isinstance(atoms_obj, Atoms):
                atoms_obj.charge = -1
                atoms_obj.set_initial_charges(np.full(len(atoms_obj), -1 / len(atoms_obj)))
            ligand_info_arg.append((atoms_obj, lig["count"]))

        cluster = build_cluster(ion_identifier=ion, ligand_molecule_info=ligand_info_arg, **cluster_kwargs)
        cluster.info["charge"] = item["charge"]
        cluster.info["category"] = item["category"]

        ion_symbol = "".join(c for c in ion if c.isalpha())
        for atom in cluster:
            if atom.symbol == ion_symbol:
                atom.charge = 1.0
                break

        write(str(xyz_dir / fname), cluster)

        kvp: Dict = {
            "category": item["category"],
            "ion": ion,
            "charge": item["charge"],
            "total_coord": item["total_coord"],
            "n_atoms": len(cluster),
            "filename": fname,
            "mix_type": item.get("mix_type", "unknown"),
            "repeat_idx": item.get("repeat_idx", 0),
            "n_solvent": item["n_solvent_total"],
            "n_anion": item["n_anion_total"],
        }

        for i, lig in enumerate(item["ligands"]):
            kvp[f"lig_{i}_name"] = lig["entry"]["name"]
            kvp[f"lig_{i}_type"] = lig["type"]
            kvp[f"lig_{i}_count"] = lig["count"]
            kvp[f"lig_{i}_src_id"] = lig["entry"].get("id", None)

            if lig["type"] == "solvent":
                lig_smiles = atom_2_smile(lig["entry"]["atoms"])
                kvp[f"lig_{i}_smiles"] = lig_smiles
                kvp[f"lig_{i}_inchi"] = smile_to_inchi(lig_smiles)

        return True, (cluster, kvp), None
    except Exception as e:
        return False, None, str(e)


def build_from_plan(
    plan: List[Dict],
    out_dir: Path,
    ion: str,
    cluster_kwargs: Dict,
    show_progress: bool,
    n_jobs: int = 32,
) -> Dict[str, int]:
    stats = {"attempted": 0, "built": 0, "failed": 0}
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "structures.db"
    if db_path.exists():
        os.remove(db_path)
    db = connect(db_path)

    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker_build_task, item, ion, xyz_dir, cluster_kwargs): item for item in plan}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(plan), desc="Building Clusters", unit="item")

        for future in iterator:
            stats["attempted"] += 1
            success, data, error = future.result()
            if success:
                cluster_obj, kvp = data
                db.write(cluster_obj, data=kvp, **kvp)
                stats["built"] += 1
            else:
                stats["failed"] += 1
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Dielectric‑constant: ref‑DB loading & matching
# ═══════════════════════════════════════════════════════════════════════════════
def load_ref_monomer_db(ref_db_path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    if not os.path.exists(ref_db_path):
        raise FileNotFoundError(f"ref_db_path not found: {ref_db_path}")

    ref_rows: List[Dict] = []
    inchi_map: Dict[str, Dict] = {}

    with connect(ref_db_path) as refdb:
        if refdb.count() == 0:
            raise ValueError(f"Reference db is empty: {ref_db_path}")

        for r in refdb.select():
            kv = dict(r.key_value_pairs)
            smiles = kv.get("Smiles") or kv.get("smiles")
            inchi = getattr(r, "inchi", None)
            dc = getattr(r, "dielectric_constant", None)
            fp = np.array(r.data["maccs_fp"]).astype(int)

            rec = {
                "id": r.id,
                "name": getattr(r, "name", None) or kv.get("name"),
                "smiles": smiles,
                "inchi": inchi,
                "dielectric_constant": dc,
                "fp": fp,
                "key_value_pairs": kv,
            }
            ref_rows.append(rec)
            if inchi:
                inchi_map[inchi] = rec

    return ref_rows, inchi_map


def best_match_by_inchi_or_fp(
    inchi: str,
    smiles: str,
    ref_rows: List[Dict],
    inchi_map: Dict[str, Dict],
) -> Dict:
    if not ref_rows:
        raise ValueError("ref_rows is empty; cannot match.")

    if inchi in inchi_map:
        return {"matched_by": "inchi_exact", "similarity": 1.0, "ref": inchi_map[inchi]}

    q_fp = smile_to_maccs_fp_arr(smiles)
    best, best_sim = None, -1.0
    for rec in ref_rows:
        sim = tanimoto_similarity(q_fp, rec["fp"])
        if sim > best_sim:
            best_sim = sim
            best = rec

    if best is None:
        raise ValueError(f"No match found for smiles={smiles}, inchi={inchi}")

    return {"matched_by": "fp_tanimoto_best", "similarity": float(best_sim), "ref": best}


# ═══════════════════════════════════════════════════════════════════════════════
# Dielectric‑constant: resolve per‑solvent ε + pure‑anion fallback
# ═══════════════════════════════════════════════════════════════════════════════
def resolve_solvent_dielectric_constants(
    solvent_pool: List[Dict],
    ref_db_path: str,
    states: List[Tuple[int, int]],
) -> Optional[float]:
    """
    Look up dielectric constant (ε) for every solvent in the pool from *ref_db_path*
    and print a detailed report so the user knows how each ε was determined.

    If any requested state has ``n_solvent == 0`` (pure‑anion cluster), computes a
    fallback ε = arithmetic mean of all resolved solvent ε values.

    Returns
    -------
    fallback_dc : float or None
        The fallback ε for pure‑anion clusters, or *None* if no such states exist.
    """
    has_pure_anion = any(n_s == 0 for n_s, _ in states)

    if not solvent_pool or not ref_db_path or not os.path.exists(ref_db_path):
        if has_pure_anion:
            print("\n  ⚠  Pure‑anion states requested but ref DB is unavailable for ε lookup.")
        return None

    try:
        ref_rows, inchi_map = load_ref_monomer_db(ref_db_path)
    except Exception as e:
        print(f"\n  ⚠  Cannot load ref DB for ε resolution: {e}")
        return None

    dc_values: List[float] = []

    print("\n" + "=" * 78)
    print("  Solvent Dielectric Constant (ε) Resolution")
    print("  Method: each solvent is matched to the ref DB via InChI or MACCS fingerprint;")
    print("          ε is then read from the matched reference entry.")
    print("-" * 78)

    for s in solvent_pool:
        atoms_obj = s["atoms"]
        name = s["name"]
        try:
            smiles = atom_2_smile(atoms_obj) if isinstance(atoms_obj, Atoms) else str(atoms_obj)
            inchi = smile_to_inchi(smiles)
            match = best_match_by_inchi_or_fp(inchi, smiles, ref_rows, inchi_map)
            ref = match["ref"]
            dc = ref.get("dielectric_constant")

            if dc is not None:
                dc_values.append(float(dc))
                print(
                    f"    {name:<22s} → ε = {dc:<8.2f}  "
                    f"(ref='{ref['name']}', {match['matched_by']}, sim={match['similarity']:.3f})"
                )
            else:
                print(
                    f"    {name:<22s} → ε = N/A      "
                    f"(ref='{ref['name']}' has no ε value)"
                )
        except Exception as e:
            print(f"    {name:<22s} → ε = ERROR    ({e})")

    fallback_dc: Optional[float] = None

    if has_pure_anion:
        print("-" * 78)
        if dc_values:
            fallback_dc = sum(dc_values) / len(dc_values)
            print(f"  ⚠  Pure‑anion states detected (n_solvent = 0).")
            print(f"     These clusters contain no solvent → ε cannot be derived from composition.")
            print(f"     Fallback ε = mean of all input solvents = {fallback_dc:.4f}")
        else:
            print(f"  ⚠  Pure‑anion states detected but no valid solvent ε was resolved.")
            print(f"     These clusters will have NO ε annotation.")

    print("=" * 78)
    return fallback_dc


# ═══════════════════════════════════════════════════════════════════════════════
# Dielectric‑constant: annotate built structures DB
# ═══════════════════════════════════════════════════════════════════════════════
def annotate_db_with_weighted_dc(
    opt_db_path: str,
    ref_db_path: str,
    fallback_dc: Optional[float] = None,
    show_progress: bool = True,
) -> None:
    """
    Walk every row in *opt_db_path*, compute a count‑weighted ε from its solvent
    ligands, and write back ``dielectric_constant`` + detailed provenance.

    For pure‑anion rows (0 solvent molecules) the *fallback_dc* is used instead.
    """
    ref_rows, inchi_map = load_ref_monomer_db(ref_db_path)

    with connect(opt_db_path) as db:
        total = db.count()
        rows = db.select()
        if show_progress:
            rows = tqdm(rows, total=total, desc="Annotating dielectric_constant", unit="row")

        for row in rows:
            kv = dict(row.key_value_pairs)
            data = dict(row.data) if row.data else {}

            def _get(key: str, _kv=kv, _data=data):
                return _kv.get(key, _data.get(key))

            ligands: list = []
            i = 0
            while _get(f"lig_{i}_type") is not None:
                if _get(f"lig_{i}_type") == "solvent":
                    ligands.append(
                        {
                            "i": i,
                            "name": _get(f"lig_{i}_name"),
                            "count": int(_get(f"lig_{i}_count") or 0),
                            "smiles": _get(f"lig_{i}_smiles"),
                            "inchi": _get(f"lig_{i}_inchi"),
                            "src_id": _get(f"lig_{i}_src_id"),
                        }
                    )
                i += 1

            total_count = sum(lg["count"] for lg in ligands)

            # ── Pure‑anion row: use fallback ε ────────────────────────────
            if total_count == 0:
                if fallback_dc is not None:
                    data["dielectric_constant_weighted_detail"] = {
                        "ref_db": ref_db_path,
                        "dielectric_constant_weighted": float(fallback_dc),
                        "components": [],
                        "weighting": "fallback_mean_of_all_input_solvents",
                        "note": "Pure‑anion cluster (n_solvent=0); ε = mean of all input solvents.",
                    }
                    db.update(row.id, dielectric_constant=float(fallback_dc), data=data)
                continue

            # ── Normal weighted ε ─────────────────────────────────────────
            dc_weighted = 0.0
            components = []

            for lg in ligands:
                smiles = lg["smiles"]
                if smiles is None:
                    raise ValueError(f"Missing lig_{lg['i']}_smiles in row id={row.id}.")

                inchi = lg["inchi"] or smile_to_inchi(smiles)
                m = best_match_by_inchi_or_fp(inchi, smiles, ref_rows, inchi_map)
                ref = m["ref"]

                if ref is None:
                    raise ValueError(f"Match ref is None for smiles={smiles}")
                if ref["dielectric_constant"] is None:
                    raise ValueError(f"No dielectric_constant: ref_id={ref['id']}")

                w = lg["count"] / total_count
                dc = float(ref["dielectric_constant"])
                dc_weighted += w * dc

                components.append(
                    {
                        "ligand": {
                            "name": lg["name"],
                            "count": lg["count"],
                            "weight": w,
                            "smiles": smiles,
                            "inchi": inchi,
                            "src_id": lg["src_id"],
                        },
                        "match": {
                            "matched_by": m["matched_by"],
                            "similarity": m["similarity"],
                            "ref_id": ref["id"],
                            "ref_name": ref["name"],
                            "ref_smiles": ref["smiles"],
                            "ref_inchi": ref["inchi"],
                            "ref_dielectric_constant": ref["dielectric_constant"],
                            "ref_key_value_pairs": ref["key_value_pairs"],
                        },
                    }
                )

            data["dielectric_constant_weighted_detail"] = {
                "ref_db": ref_db_path,
                "dielectric_constant_weighted": float(dc_weighted),
                "components": components,
                "weighting": "solvent_count_fraction",
            }
            db.update(row.id, dielectric_constant=float(dc_weighted), data=data)


# ═══════════════════════════════════════════════════════════════════════════════
# Printing helpers
# ═══════════════════════════════════════════════════════════════════════════════
def print_monomer_sources(solvents: List[Dict], anions: List[Dict]) -> None:
    print("\n" + "=" * 70)
    print("  Monomer Source Summary")
    print("=" * 70)

    if solvents:
        print(f"\n  Solvents ({len(solvents)}):")
        for s in solvents:
            src = s.get("source", "unknown")
            print(f"    • {s['name']:<25s}  ←  {src}")
    else:
        print("\n  Solvents: (none)")

    if anions:
        print(f"\n  Anions ({len(anions)}):")
        for a in anions:
            src = a.get("source", "unknown")
            print(f"    • {a['name']:<25s}  ←  {src}")
    else:
        print("\n  Anions: (none)")

    print("=" * 70)


def print_plan_summary(full_plan: List[Dict]) -> None:
    total_tasks = len(full_plan)

    state_map: Dict[Tuple[int, int], List[Dict]] = {}
    for task in full_plan:
        key = (task["n_solvent_total"], task["n_anion_total"])
        state_map.setdefault(key, []).append(task)

    print("\n" + "=" * 70)
    print("  Generation Plan Summary")
    print(f"  Total Tasks: {total_tasks}")
    print("-" * 70)

    for key in sorted(state_map, key=lambda x: (x[0] + x[1], x[1])):
        n_solv, n_ani = key
        cat = "SSIP" if n_ani == 0 else ("CIP" if n_ani == 1 else "AGG")
        count = len(state_map[key])
        pct = (count / total_tasks) * 100
        print(f"  State {n_solv:>2} Solv : {n_ani:>2} Anion ({cat:<4}) | {count:>6} tasks | {pct:>5.1f}%")

    print("-" * 70)
    print("  Preview (up to 5 per state)")

    for key in sorted(state_map, key=lambda x: (x[0] + x[1], x[1])):
        n_solv, n_ani = key
        cat = "SSIP" if n_ani == 0 else ("CIP" if n_ani == 1 else "AGG")
        items = state_map[key]
        print(f"\n  [State: {n_solv} Solv : {n_ani} Anion ({cat})]")
        for i, task in enumerate(random.sample(items, min(5, len(items)))):
            ligand_strs = [f"{lg['count']}x {lg['entry']['name']} ({lg['type']})" for lg in task["ligands"]]
            print(f"    {i + 1}. Coord={task['total_coord']} | {' + '.join(ligand_strs)} | Run #{task['repeat_idx']}")

    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Default cluster kwargs
# ═══════════════════════════════════════════════════════════════════════════════
def _default_cluster_kwargs(**overrides) -> Dict:
    defaults = dict(
        relative_score_threshold=0.8,
        max_patch_atoms=2,
        initial_sphere_skin_factor=0.7,
        sphere_skin_increment_factor=0.01,
        target_no_clashes=True,
        rotation_opt_iterations=50,
        max_sphere_expansions=100,
        verbose=False,
    )
    defaults.update(overrides)
    return defaults


# ═══════════════════════════════════════════════════════════════════════════════
# Post‑build UMA optimisation + DC annotation
# ═══════════════════════════════════════════════════════════════════════════════
def postprocess(
    raw_db_path: str,
    ref_db_path: str,
    out_path: Path,
    use_uma: bool,
    device: str,
    verbose: bool,
    show_progress: bool,
    n_built: int,
    fallback_dc: Optional[float] = None,
) -> None:
    target_db = raw_db_path
    if use_uma and UMA_AVAILABLE and n_built > 0:
        opt_dir = out_path / "optimized"
        print(f"\nRunning UMA Optimisation on {raw_db_path} ...")
        target_db = uma_entry.entry(
            input_db=raw_db_path,
            workspace=str(opt_dir),
            device=device,
            verbose=verbose,
            show_progress=show_progress,
        )
    annotate_db_with_weighted_dc(target_db, ref_db_path, fallback_dc, show_progress)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════════
def entry(
    solvents: Union[str, List[str]],
    anions: Union[str, List[str]],
    ref_db_path: str = DEFAULT_REF_DB_PATH,
    out_dir: str = "out_mixture",
    ion: str = "Li",
    states: List[Tuple[int, int]] = [(3, 1), (4, 1)],
    solvent_mix_sizes: Tuple[int, ...] = (1,),
    anion_mix_sizes: Tuple[int, ...] = (1,),
    repeats: int = 1,
    use_uma: bool = True,
    device: str = "cuda",
    verbose: bool = True,
    show_progress: bool = True,
    n_jobs: int = 32,
    **cluster_kwargs,
):
    """
    End‑to‑end pipeline: resolve monomers → plan mixtures → build clusters →
    (optional) UMA optimise → annotate dielectric constants.

    Parameters
    ----------
    solvent_mix_sizes : tuple of int
        Number of *distinct* solvent species per cluster.
        ``(1,)`` = single‑solvent; ``(1, 2)`` = also try binary mixtures.
    anion_mix_sizes : tuple of int
        Number of *distinct* anion species per cluster.
    states : list of (n_solvent, n_anion)
        Each pair gives the molecule count of solvent and anion around the ion.
    """
    # ── 1. Resolve monomers (ref‑DB lookup → UMA / fallback) ──────────────
    solv_data = normalize_input_data(
        solvents, "Solvent", show_progress, out_dir, device, use_uma,
        ref_db_for_lookup=ref_db_path,
    )
    anion_data = normalize_input_data(
        anions, "Anion", show_progress, out_dir, device, use_uma,
        ref_db_for_lookup=_ANION_REF_DB_PATH,
    )

    if not solv_data:
        raise ValueError("No solvent data found.")

    # ── 2. Show where every monomer came from ─────────────────────────────
    print_monomer_sources(solv_data, anion_data)

    # ── 3. Resolve per‑solvent ε & pure‑anion fallback ────────────────────
    fallback_dc = resolve_solvent_dielectric_constants(solv_data, ref_db_path, states)

    # ── 4. Plan mixtures ──────────────────────────────────────────────────
    full_plan = plan_mixtures(
        solvents_pool=solv_data,
        anions_pool=anion_data,
        solvent_mix_sizes=solvent_mix_sizes,
        anion_mix_sizes=anion_mix_sizes,
        states=states,
        repeats=repeats,
    )

    if not full_plan:
        print("Plan is empty. Check constraints.")
        return

    print_plan_summary(full_plan)

    # ── 5. Build clusters ─────────────────────────────────────────────────
    out_path = Path(out_dir)
    final_kwargs = _default_cluster_kwargs(**cluster_kwargs)
    stats = build_from_plan(full_plan, out_path, ion, final_kwargs, show_progress, n_jobs=n_jobs)
    print(f"\nBuild Done: {stats['built']}/{stats['attempted']} success.")

    # ── 6. Post‑build UMA optimisation + ε annotation ─────────────────────
    raw_db = str(out_path / "structures.db")
    postprocess(
        raw_db, ref_db_path, out_path,
        use_uma, device, verbose, show_progress,
        stats["built"], fallback_dc,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster factory: build mixtures + (optional) UMA optimize")
    parser.add_argument("--solvents", type=str, default=DEFAULT_DME_SMILES, help="SMILES or db path")
    parser.add_argument("--anions", type=str, default=DEFAULT_FSI_SMILES, help="SMILES or db path")
    parser.add_argument(
        "--ref_db_path", type=str, default=DEFAULT_REF_DB_PATH,
        help="Solvent ref db (must contain: inchi, dielectric_constant, data['maccs_fp'])",
    )
    parser.add_argument("--out_dir", type=str, default="out_mixture")
    parser.add_argument("--ion", type=str, default="Li")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--use_uma", action="store_false", default=True)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()

    entry(
        solvents=args.solvents,
        anions=args.anions,
        ref_db_path=args.ref_db_path,
        out_dir=args.out_dir,
        ion=args.ion,
        repeats=args.repeats,
        use_uma=args.use_uma,
        device=args.device,
        n_jobs=args.n_jobs,
        show_progress=True,
        verbose=True,
    )