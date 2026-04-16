from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from functools import lru_cache

import numpy as np
from ase.db import connect
from ase.neighborlist import NeighborList, natural_cutoffs

DEFAULT_SOLVENT_REF_DB_PATH = "/home/mingkang_nt/hetero_atoms_workspace/dc_0314_build/sol_w_dc.db"
DEFAULT_ANION_REF_DB_PATH = "/home/mingkang_nt/hetero_atoms_workspace/dc_0314_build/anion.db"
DEFAULT_CATION_SYMBOLS = ("Li",)

_PCT_LABELS = ("anion", "solvent", "cation", "unknown")


def _normalize_name(value):
    if value is None:
        return None
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _name_aliases(value):
    if value is None:
        return set()
    raw = str(value).strip()
    aliases = {_normalize_name(raw)}
    for token in re.split(r"[\s(/,;-]+", raw):
        token_norm = _normalize_name(token)
        if token_norm:
            aliases.add(token_norm)
    return {item for item in aliases if item}


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _merged_row_meta(row):
    merged = {}
    merged.update(dict(getattr(row, "key_value_pairs", None) or {}))
    merged.update(dict(getattr(row, "data", None) or {}))
    return merged


def _element_signature(symbols):
    return tuple(sorted(Counter(symbols).items()))


def _atoms_descriptor(atoms, atom_indices=None):
    if atom_indices is None:
        symbols = list(atoms.get_chemical_symbols())
    else:
        symbols = [atoms[int(idx)].symbol for idx in atom_indices]
    numbers = [1 if symbol == "H" else 0 for symbol in symbols]
    return {
        "natoms": len(symbols),
        "heavy_atoms": int(len(symbols) - sum(numbers)),
        "element_signature": _element_signature(symbols),
        "formula": "".join(
            f"{sym}{count if count > 1 else ''}" for sym, count in _element_signature(symbols)
        ),
    }


def _parse_ligands_from_row(row):
    meta = _merged_row_meta(row)
    ligands = []
    lig_idx = 0
    while meta.get(f"lig_{lig_idx}_type") is not None:
        ligands.append(
            {
                "index": lig_idx,
                "type": str(meta.get(f"lig_{lig_idx}_type")).strip().lower(),
                "name": meta.get(f"lig_{lig_idx}_name"),
                "count": _safe_int(meta.get(f"lig_{lig_idx}_count"), 0) or 0,
                "src_id": _safe_int(meta.get(f"lig_{lig_idx}_src_id"), None),
                "smiles": meta.get(f"lig_{lig_idx}_smiles"),
                "inchi": meta.get(f"lig_{lig_idx}_inchi"),
            }
        )
        lig_idx += 1
    return meta, ligands


def _parse_filename_counts(filename):
    counts = []
    if not filename:
        return counts
    for family, name, count in re.findall(r"([SA])\d+-([^_]+?)_n(\d+)", str(filename)):
        counts.append(
            {
                "type": "solvent" if family == "S" else "anion",
                "name": name,
                "count": int(count),
            }
        )
    return counts


def _build_count_checks(filename_counts, ligands):
    return {
        "filename_count_check": {
            f"{item['type']}::{_normalize_name(item['name'])}": int(item["count"])
            for item in filename_counts
        },
        "meta_count_check": {
            f"{lig['type']}::{_normalize_name(lig['name'])}": int(lig["count"])
            for lig in ligands
        },
    }


def _skip_fragment_result(reason, filename_counts, ligands):
    return {
        "skipped": True,
        "skip_reason": reason,
        **_build_count_checks(filename_counts=filename_counts, ligands=ligands),
    }


def _match_ligand_identity(ligand, candidate_name=None, candidate_smiles=None, candidate_inchi=None):
    ligand_aliases = _name_aliases(ligand.get("name"))
    candidate_aliases = _name_aliases(candidate_name)
    if ligand_aliases and candidate_aliases and ligand_aliases.intersection(candidate_aliases):
        return True

    ligand_smiles = ligand.get("smiles")
    if ligand_smiles and candidate_smiles and str(ligand_smiles).strip() == str(candidate_smiles).strip():
        return True

    ligand_inchi = ligand.get("inchi")
    if ligand_inchi and candidate_inchi and str(ligand_inchi).strip() == str(candidate_inchi).strip():
        return True

    return False


@lru_cache(maxsize=4)
def _load_reference_index(db_path):
    index = {
        "rows": [],
        "by_id": {},
        "by_ordinal_idx": {},
        "by_name": {},
        "by_signature": {},
    }
    if not db_path or not os.path.exists(db_path):
        return index

    with connect(db_path) as db:
        for ordinal_idx, row in enumerate(db.select()):
            atoms = row.toatoms()
            kv = dict(getattr(row, "key_value_pairs", None) or {})
            data = dict(getattr(row, "data", None) or {})
            name = getattr(row, "name", None) or kv.get("name") or kv.get("Name")
            desc = _atoms_descriptor(atoms)
            record = {
                "id": int(row.id),
                "ordinal_idx": int(ordinal_idx),
                "name": name,
                "name_aliases": sorted(_name_aliases(name)),
                "natoms": desc["natoms"],
                "heavy_atoms": desc["heavy_atoms"],
                "element_signature": desc["element_signature"],
                "formula": atoms.get_chemical_formula(),
                "smiles": kv.get("smiles") or kv.get("Smiles") or data.get("smiles") or data.get("Smiles"),
                "inchi": kv.get("inchi") or kv.get("InChI") or data.get("inchi") or data.get("InChI"),
            }
            index["rows"].append(record)
            index["by_id"][record["id"]] = record
            index["by_ordinal_idx"][record["ordinal_idx"]] = record
            for alias in record["name_aliases"]:
                index["by_name"].setdefault(alias, []).append(record)
            index["by_signature"].setdefault(
                (record["natoms"], record["element_signature"]),
                [],
            ).append(record)
    return index


def _resolve_solvent_record_from_dielectric_detail(ligand, meta, ref_index):
    detail = meta.get("dielectric_constant_weighted_detail") or {}
    for component in detail.get("components") or []:
        ligand_info = component.get("ligand") or {}
        match_info = component.get("match") or {}
        if not _match_ligand_identity(
            ligand,
            candidate_name=ligand_info.get("name") or match_info.get("ref_name"),
            candidate_smiles=ligand_info.get("smiles") or match_info.get("ref_smiles"),
            candidate_inchi=ligand_info.get("inchi") or match_info.get("ref_inchi"),
        ):
            continue

        ref_id = _safe_int(match_info.get("ref_id"), None)
        if ref_id is not None and ref_id in ref_index["by_id"]:
            return ref_index["by_id"][ref_id]

        for alias in _name_aliases(match_info.get("ref_name")):
            candidates = ref_index["by_name"].get(alias, [])
            if len(candidates) == 1:
                return candidates[0]

    return None


def _resolve_reference_record(ligand, meta, solvent_ref_db_path, anion_ref_db_path):
    db_path = anion_ref_db_path if ligand["type"] == "anion" else solvent_ref_db_path
    ref_index = _load_reference_index(db_path)
    src_id = ligand.get("src_id")
    ligand_aliases = _name_aliases(ligand.get("name"))

    if src_id is not None:
        src_candidates = []
        if src_id in ref_index["by_ordinal_idx"]:
            src_candidates.append(ref_index["by_ordinal_idx"][src_id])
        if src_id in ref_index["by_id"]:
            src_candidates.append(ref_index["by_id"][src_id])

        if src_candidates:
            for candidate in src_candidates:
                if ligand_aliases.intersection(candidate.get("name_aliases", [])):
                    return candidate
            if len(src_candidates) == 1:
                return src_candidates[0]

    if ligand["type"] == "solvent":
        resolved = _resolve_solvent_record_from_dielectric_detail(ligand, meta=meta, ref_index=ref_index)
        if resolved is not None:
            return resolved

    for name_alias in ligand_aliases:
        if name_alias in ref_index["by_name"]:
            candidates = ref_index["by_name"][name_alias]
            if len(candidates) == 1:
                return candidates[0]

    return None


def _component_indices_without_cations(atoms, connectivity_mult=1.1, cation_symbols=DEFAULT_CATION_SYMBOLS):
    cation_indices = [idx for idx, symbol in enumerate(atoms.get_chemical_symbols()) if symbol in cation_symbols]
    non_cation_indices = [idx for idx in range(len(atoms)) if idx not in cation_indices]
    if not non_cation_indices:
        return [], cation_indices

    sub_atoms = atoms[non_cation_indices]
    cutoffs = natural_cutoffs(sub_atoms, mult=float(connectivity_mult))
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(sub_atoms)

    adjacency = [set() for _ in range(len(sub_atoms))]
    for i in range(len(sub_atoms)):
        neighbors, _ = nl.get_neighbors(i)
        for j in neighbors:
            adjacency[i].add(int(j))
            adjacency[int(j)].add(i)

    visited = [False] * len(sub_atoms)
    components = []
    for start in range(len(sub_atoms)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        local = []
        while stack:
            node = stack.pop()
            local.append(non_cation_indices[node])
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(sorted(local))
    return components, cation_indices


def _match_components_to_ligands(
    components,
    component_descs,
    ligand_specs,
    remaining_component_indices=None,
):
    remaining = set(
        range(len(components)) if remaining_component_indices is None else remaining_component_indices
    )
    assignments = {}

    expanded_specs = []
    for lig in ligand_specs:
        for instance_idx in range(max(0, int(lig.get("count", 0)))):
            expanded_specs.append(
                {
                    "type": lig["type"],
                    "name": lig.get("name"),
                    "ligand_index": lig["index"],
                    "instance_index": instance_idx,
                    "ref": lig.get("ref"),
                }
            )

    def _spec_sort_key(spec):
        ref = spec.get("ref") or {}
        type_rank = 0 if spec["type"] == "anion" else 1
        return (
            type_rank,
            -int(ref.get("natoms", 0)),
            -int(ref.get("heavy_atoms", 0)),
            spec.get("ligand_index", 0),
            spec.get("instance_index", 0),
        )

    expanded_specs.sort(key=_spec_sort_key)

    for spec in expanded_specs:
        ref = spec.get("ref")
        if ref is None:
            continue

        exact = []
        fallback = []
        for comp_idx in sorted(remaining):
            desc = component_descs[comp_idx]
            if desc["natoms"] == ref["natoms"] and desc["element_signature"] == ref["element_signature"]:
                exact.append(comp_idx)
            elif desc["heavy_atoms"] == ref["heavy_atoms"] and desc["element_signature"] == ref["element_signature"]:
                fallback.append(comp_idx)

        chosen = exact[0] if exact else (fallback[0] if fallback else None)
        if chosen is None:
            continue
        assignments[chosen] = spec
        remaining.remove(chosen)

    return assignments


def _assign_remaining_components_as_solvent(assignments, components, ligand_specs):
    remaining_components = sorted(set(range(len(components))) - set(assignments))
    if not remaining_components:
        return assignments

    solvent_ligands = [lig for lig in ligand_specs if lig["type"] == "solvent"]
    assigned_by_family = Counter(
        int(spec.get("ligand_index", -1))
        for spec in assignments.values()
        if spec.get("type") == "solvent"
    )

    default_ligand = solvent_ligands[0] if len(solvent_ligands) == 1 else None
    for instance_offset, comp_idx in enumerate(remaining_components):
        assignments[comp_idx] = {
            "type": "solvent",
            "name": None if default_ligand is None else default_ligand.get("name"),
            "ligand_index": -1 if default_ligand is None else int(default_ligand["index"]),
            "instance_index": (
                int(instance_offset)
                if default_ligand is None
                else int(assigned_by_family.get(default_ligand["index"], 0) + instance_offset)
            ),
            "ref": None if default_ligand is None else default_ligand.get("ref"),
            "assignment_mode": "assume_remaining_components_are_solvent",
        }
    return assignments


def infer_fragment_labels(
    row,
    atoms,
    connectivity_mult=1.1,
    solvent_ref_db_path=DEFAULT_SOLVENT_REF_DB_PATH,
    anion_ref_db_path=DEFAULT_ANION_REF_DB_PATH,
    cation_symbols=DEFAULT_CATION_SYMBOLS,
):
    meta, ligands = _parse_ligands_from_row(row)
    filename_counts = _parse_filename_counts(meta.get("filename"))
    for lig in ligands:
        lig["ref"] = _resolve_reference_record(
            ligand=lig,
            meta=meta,
            solvent_ref_db_path=solvent_ref_db_path,
            anion_ref_db_path=anion_ref_db_path,
        )

    anion_ligands = [lig for lig in ligands if lig["type"] == "anion"]
    unresolved_anions = [lig for lig in anion_ligands if lig.get("ref") is None]
    if unresolved_anions:
        unresolved_names = [lig.get("name") or f"lig_{lig['index']}" for lig in unresolved_anions]
        reason = f"anion references not found: {', '.join(unresolved_names)}"
        warnings.warn(f"[fragment_hosting] {reason}; skip fragment hosting.")
        return _skip_fragment_result(reason, filename_counts=filename_counts, ligands=ligands)

    components, cation_indices = _component_indices_without_cations(
        atoms=atoms,
        connectivity_mult=connectivity_mult,
        cation_symbols=cation_symbols,
    )
    component_descs = [_atoms_descriptor(atoms, atom_indices=comp) for comp in components]
    assignments = _match_components_to_ligands(
        components=components,
        component_descs=component_descs,
        ligand_specs=anion_ligands,
    )
    expected_anion_components = sum(int(lig.get("count", 0)) for lig in anion_ligands)
    if len(assignments) != expected_anion_components:
        reason = (
            f"anion component matching incomplete: expected {expected_anion_components}, "
            f"matched {len(assignments)}"
        )
        warnings.warn(f"[fragment_hosting] {reason}; skip fragment hosting.")
        return _skip_fragment_result(reason, filename_counts=filename_counts, ligands=ligands)

    assignments.update(
        _match_components_to_ligands(
            components=components,
            component_descs=component_descs,
            ligand_specs=[lig for lig in ligands if lig["type"] == "solvent" and lig.get("ref") is not None],
            remaining_component_indices=sorted(set(range(len(components))) - set(assignments)),
        )
    )
    assignments = _assign_remaining_components_as_solvent(
        assignments=assignments,
        components=components,
        ligand_specs=ligands,
    )

    atom_labels = np.array(["unknown"] * len(atoms), dtype=object)
    for atom_idx in cation_indices:
        atom_labels[int(atom_idx)] = "cation"

    component_info = []
    for comp_idx, atom_indices in enumerate(components):
        spec = assignments.get(comp_idx)
        label = spec["type"] if spec is not None else "unknown"
        for atom_idx in atom_indices:
            atom_labels[int(atom_idx)] = label
        info = {
            "component_index": int(comp_idx),
            "atom_indices": [int(x) for x in atom_indices],
            "natoms": int(component_descs[comp_idx]["natoms"]),
            "heavy_atoms": int(component_descs[comp_idx]["heavy_atoms"]),
            "element_signature": list(component_descs[comp_idx]["element_signature"]),
            "assigned_label": label,
        }
        if spec is not None:
            info.update(
                {
                    "assigned_name": spec.get("name"),
                    "ligand_index": int(spec.get("ligand_index", -1)),
                    "instance_index": int(spec.get("instance_index", -1)),
                    "reference_id": None if spec.get("ref") is None else int(spec["ref"]["id"]),
                    "assignment_mode": spec.get("assignment_mode", "reference_signature"),
                }
            )
        component_info.append(info)

    counts_by_label = Counter(atom_labels.tolist())
    if counts_by_label.get("unknown", 0):
        reason = f"unknown fragment labels remain: {counts_by_label.get('unknown', 0)} atoms"
        warnings.warn(f"[fragment_hosting] {reason}; skip fragment hosting.")
        return _skip_fragment_result(reason, filename_counts=filename_counts, ligands=ligands)

    assignment_coverage = float(100.0 * np.mean(atom_labels != "unknown")) if len(atom_labels) else 0.0
    count_checks = _build_count_checks(filename_counts=filename_counts, ligands=ligands)

    return {
        "skipped": False,
        "atom_labels": atom_labels,
        "component_info": component_info,
        "assignment_coverage_pct": assignment_coverage,
        "counts_by_label": {key: int(val) for key, val in sorted(counts_by_label.items())},
        **count_checks,
    }


def _ao_to_atom_indices(mol):
    ao_to_atom = []
    for atom_idx, (_, _, ao_start, ao_stop) in enumerate(mol.aoslice_by_atom()):
        ao_to_atom.extend([atom_idx] * int(ao_stop - ao_start))
    return np.asarray(ao_to_atom, dtype=np.int32)


def _lowdin_orbital_weights(coefficients, overlap):
    coeff = np.asarray(coefficients, dtype=float).reshape(-1)
    ov = overlap[0] if getattr(overlap, "ndim", 0) == 3 else np.asarray(overlap, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(ov)
    eigvals = np.clip(eigvals, 1e-12, None)
    ov_half = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    coeff_orth = ov_half @ coeff
    weights = np.abs(coeff_orth) ** 2
    total = float(np.sum(weights))
    if total > 0:
        weights = weights / total
    return weights


def calculate_fragment_hosting_percentages(mol, overlap, orbital_coefficients, atom_labels):
    ao_to_atom = _ao_to_atom_indices(mol)
    atom_labels = np.asarray(atom_labels, dtype=object)
    ao_weights = _lowdin_orbital_weights(orbital_coefficients, overlap)
    percentages = {}
    for label in _PCT_LABELS:
        mask = atom_labels[ao_to_atom] == label
        percentages[label] = float(100.0 * np.sum(ao_weights[mask]))
    percentages["host_label"] = max(_PCT_LABELS, key=lambda item: percentages[item])
    return percentages


def infer_orbital_fragment_hosting(
    row,
    atoms,
    mol,
    overlap,
    homo_coefficients=None,
    lumo_coefficients=None,
    connectivity_mult=1.1,
    solvent_ref_db_path=DEFAULT_SOLVENT_REF_DB_PATH,
    anion_ref_db_path=DEFAULT_ANION_REF_DB_PATH,
):
    fragment_info = infer_fragment_labels(
        row=row,
        atoms=atoms,
        connectivity_mult=connectivity_mult,
        solvent_ref_db_path=solvent_ref_db_path,
        anion_ref_db_path=anion_ref_db_path,
    )
    if fragment_info.get("skipped"):
        return {
            "fragment_hosting_skipped": True,
            "fragment_hosting_skip_reason": fragment_info.get("skip_reason"),
            "fragment_filename_count_check": fragment_info.get("filename_count_check", {}),
            "fragment_meta_count_check": fragment_info.get("meta_count_check", {}),
        }

    atom_labels = fragment_info["atom_labels"]

    result = {
        "fragment_hosting_skipped": False,
        "fragment_assignment_coverage_pct": float(fragment_info["assignment_coverage_pct"]),
        "fragment_atom_count_anion": int(fragment_info["counts_by_label"].get("anion", 0)),
        "fragment_atom_count_solvent": int(fragment_info["counts_by_label"].get("solvent", 0)),
        "fragment_atom_count_cation": int(fragment_info["counts_by_label"].get("cation", 0)),
        "fragment_atom_count_unknown": int(fragment_info["counts_by_label"].get("unknown", 0)),
        "fragment_filename_count_check": fragment_info["filename_count_check"],
        "fragment_meta_count_check": fragment_info["meta_count_check"],
        "fragment_component_info": fragment_info["component_info"],
    }

    if homo_coefficients is not None:
        homo_pct = calculate_fragment_hosting_percentages(mol, overlap, homo_coefficients, atom_labels)
        result.update(
            {
                "HOMO_anion_pct": float(homo_pct["anion"]),
                "HOMO_solvent_pct": float(homo_pct["solvent"]),
                "HOMO_cation_pct": float(homo_pct["cation"]),
                "HOMO_unknown_pct": float(homo_pct["unknown"]),
                "HOMO_host_label": homo_pct["host_label"],
            }
        )

    if lumo_coefficients is not None:
        lumo_pct = calculate_fragment_hosting_percentages(mol, overlap, lumo_coefficients, atom_labels)
        result.update(
            {
                "LUMO_anion_pct": float(lumo_pct["anion"]),
                "LUMO_solvent_pct": float(lumo_pct["solvent"]),
                "LUMO_cation_pct": float(lumo_pct["cation"]),
                "LUMO_unknown_pct": float(lumo_pct["unknown"]),
                "LUMO_host_label": lumo_pct["host_label"],
            }
        )

    return result
