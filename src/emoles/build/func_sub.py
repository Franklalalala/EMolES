from __future__ import annotations

import contextlib
import io
import json
import random
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.db import connect
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import WrapLogs


# ============================================================
# 1) RDKit log capture
# ============================================================
@contextlib.contextmanager
def _rdkit_capture_stderr():
    WrapLogs()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf


@contextlib.contextmanager
def _rdkit_silent():
    WrapLogs()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield


# ============================================================
# 2) ASE Atoms -> smiles/inchi
# ============================================================
def atoms_to_xyz_block(atoms: Atoms) -> Optional[str]:
    syms = atoms.get_chemical_symbols()
    if any(s == "X" for s in syms):
        return None
    pos = atoms.get_positions()
    lines = [str(len(atoms)), "ase_atoms"]
    for s, (x, y, z) in zip(syms, pos):
        lines.append(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n"


def atoms_to_smiles_inchi_fast(atoms: Atoms) -> Tuple[Optional[str], Optional[str]]:
    xyz = atoms_to_xyz_block(atoms)
    if xyz is None:
        return None, None
    with _rdkit_silent():
        try:
            mol = Chem.MolFromXYZBlock(xyz)
            if mol is None:
                return None, None
            DetermineBonds(mol, useHueckel=True)
            Chem.SanitizeMol(mol)
            mol_no_h = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)
            inchi = Chem.MolToInchi(mol_no_h)
            return smiles, inchi
        except Exception:
            return None, None


def atoms_to_smiles_inchi_with_log(atoms: Atoms) -> Tuple[Optional[str], Optional[str], str]:
    xyz = atoms_to_xyz_block(atoms)
    if xyz is None:
        return None, None, "XYZ_BLOCK_FAIL(X present?)"
    with _rdkit_capture_stderr() as cap:
        try:
            mol = Chem.MolFromXYZBlock(xyz)
            if mol is None:
                return None, None, cap.getvalue() + "\nMolFromXYZBlock=None"
            DetermineBonds(mol, useHueckel=True)
            Chem.SanitizeMol(mol)
            mol_no_h = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol_no_h, isomericSmiles=False)
            inchi = Chem.MolToInchi(mol_no_h)
            return smiles, inchi, cap.getvalue()
        except Exception as e:
            return None, None, cap.getvalue() + f"\nEXC: {repr(e)}"


# ============================================================
# 3) Filters & Distance Helpers
# ============================================================
@dataclass(frozen=True)
class FilterConfig:
    covalent_radius_factor: float = 1.10
    min_distance_clash: float = 0.55
    require_single_component: bool = True
    forbid_isolated_atom: bool = True

    forbid_3_4_member_rings: bool = True
    forbid_cc_triple: bool = True
    forbid_oo_bond: bool = True
    forbid_oh_bond: bool = True

    reject_rdkit_warnings: bool = True


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    d = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _pairwise_distances_two(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    d = pos1[:, None, :] - pos2[None, :, :]
    return np.sqrt((d * d).sum(-1))


def _bond_graph_by_radii(atoms: Atoms, covalent_radius_factor: float) -> List[List[int]]:
    n = len(atoms)
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()

    dist = _pairwise_distances(pos) + np.eye(n) * 1e9
    r = covalent_radii[nums]
    thresh = (r[:, None] + r[None, :]) * covalent_radius_factor
    bonded = dist <= thresh
    return [list(np.where(bonded[i])[0]) for i in range(n)]


def topo_geometry_filter(atoms: Atoms, cfg: FilterConfig) -> Tuple[bool, str]:
    syms = atoms.get_chemical_symbols()
    if any(s == "X" for s in syms):
        return False, "DUMMY_PRESENT"

    n = len(atoms)
    if n < 2:
        return False, "NOT_MOLECULE"

    pos = atoms.get_positions()
    dist = _pairwise_distances(pos) + np.eye(n) * 1e9
    if float(dist.min()) < cfg.min_distance_clash:
        return False, "CLASH"

    adj = _bond_graph_by_radii(atoms, covalent_radius_factor=cfg.covalent_radius_factor)

    if cfg.forbid_isolated_atom:
        deg = np.array([len(x) for x in adj], dtype=int)
        if np.any(deg == 0):
            return False, "ISOLATED_ATOM"

    if cfg.require_single_component:
        seen = np.zeros(n, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        if not bool(seen.all()):
            return False, "DISCONNECTED"

    return True, "OK"


def smiles_rule_filter(smiles: str, cfg: FilterConfig) -> Tuple[bool, str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "BAD_SMILES"

        if cfg.forbid_3_4_member_rings:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                if len(ring) in (3, 4):
                    return False, "SMALL_RING"

        if cfg.forbid_cc_triple:
            for b in mol.GetBonds():
                if b.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                    if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
                        return False, "CC_TRIPLE"

        if cfg.forbid_oo_bond or cfg.forbid_oh_bond:
            molH = Chem.AddHs(mol)
            for b in molH.GetBonds():
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                z1, z2 = a1.GetAtomicNum(), a2.GetAtomicNum()
                if cfg.forbid_oo_bond and z1 == 8 and z2 == 8:
                    return False, "OO_BOND"
                if cfg.forbid_oh_bond and ((z1, z2) == (8, 1) or (z1, z2) == (1, 8)):
                    return False, "OH_BOND"

        return True, "OK"
    except Exception:
        return False, "SMILES_RULE_FAIL"


def _rdkit_warning_bad(log: str) -> bool:
    if not log:
        return False
    bad_keys = [
        "Charges were rearranged",
        "Accepted unusual valence",
        "Explicit valence for atom",
        "KekulizeException",
    ]
    return any(k in log for k in bad_keys)


def clean_filter_atoms(atoms: Atoms, cfg: FilterConfig) -> Tuple[bool, Optional[str], Optional[str], str]:
    ok, reason = topo_geometry_filter(atoms, cfg)
    if not ok:
        return False, None, None, reason

    if cfg.reject_rdkit_warnings:
        smiles, inchi, log = atoms_to_smiles_inchi_with_log(atoms)
        if smiles is None or inchi is None:
            return False, None, None, "RDKit_FAIL"
        if _rdkit_warning_bad(log):
            return False, None, None, "RDKit_WARN_VALENCE"
    else:
        smiles, inchi = atoms_to_smiles_inchi_fast(atoms)
        if smiles is None or inchi is None:
            return False, None, None, "RDKit_FAIL"

    ok2, reason2 = smiles_rule_filter(smiles, cfg)
    if not ok2:
        return False, None, None, reason2

    return True, smiles, inchi, "OK"


# ============================================================
# 4) Fragment library
# ============================================================
@dataclass(frozen=True)
class FragEntry:
    atoms: Atoms
    dummy2_idx: int
    attach_idx: int


def _rdkit_mol_3d(smiles: str, seed: int = 0) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad smiles: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True

    code = AllChem.EmbedMolecule(mol, params)
    if code != 0:
        code = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if code != 0:
            raise RuntimeError(f"RDKit Embed failed for: {smiles}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass

    return mol


def _rdkit_to_ase_atoms(mol: Chem.Mol) -> Atoms:
    pt = Chem.GetPeriodicTable()
    conf = mol.GetConformer()
    syms = []
    pos = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        syms.append("X" if z == 0 else pt.GetElementSymbol(z))
        p = conf.GetAtomPosition(a.GetIdx())
        pos.append([float(p.x), float(p.y), float(p.z)])
    return Atoms("".join(syms), positions=np.array(pos, dtype=float))


def frag_from_smiles_by_replacing_terminal(
    smiles: str,
    placeholder_symbol: str = "Cl",
    seed: int = 0,
) -> FragEntry:
    mol = _rdkit_mol_3d(smiles, seed=seed)

    ph_idx = None
    for a in mol.GetAtoms():
        if a.GetSymbol() == placeholder_symbol and a.GetDegree() == 1:
            ph_idx = a.GetIdx()
            break
    if ph_idx is None:
        raise ValueError(f"no terminal {placeholder_symbol} found in {smiles}")

    attach_idx = mol.GetAtomWithIdx(ph_idx).GetNeighbors()[0].GetIdx()
    mol.GetAtomWithIdx(ph_idx).SetAtomicNum(0)

    atoms = _rdkit_to_ase_atoms(mol)
    return FragEntry(atoms=atoms, dummy2_idx=int(ph_idx), attach_idx=int(attach_idx))


def make_frag_library_default(seed: int = 0) -> Dict[str, FragEntry]:
    lib: Dict[str, FragEntry] = {}
    lib["CH3"] = frag_from_smiles_by_replacing_terminal("CCl", "Cl", seed=seed)
    lib["CF3"] = frag_from_smiles_by_replacing_terminal("FC(F)(F)Cl", "Cl", seed=seed)
    lib["CN"] = frag_from_smiles_by_replacing_terminal("N#CCl", "Cl", seed=seed)
    lib["OCH3"] = frag_from_smiles_by_replacing_terminal("COCl", "Cl", seed=seed)
    lib["COOCH3"] = frag_from_smiles_by_replacing_terminal("COC(=O)Cl", "Cl", seed=seed)
    lib["SO2F"] = frag_from_smiles_by_replacing_terminal("O=S(=O)(F)Cl", "Cl", seed=seed)
    lib["F"] = frag_from_smiles_by_replacing_terminal("[H]F", "H", seed=seed)
    lib["SO2CH3"] = frag_from_smiles_by_replacing_terminal("CS(=O)(=O)Cl", "Cl", seed=seed)
    return lib


# ============================================================
# 5) Site helpers
# ============================================================
def list_all_h_indices(atoms: Atoms) -> List[int]:
    return [a.index for a in atoms if a.symbol == "H"]


def find_oh_h_indices(atoms: Atoms, covalent_radius_factor: float = 1.18) -> List[int]:
    syms = atoms.get_chemical_symbols()
    adj = _bond_graph_by_radii(atoms, covalent_radius_factor=covalent_radius_factor)

    oh = []
    for i, s in enumerate(syms):
        if s != "H":
            continue
        neigh = adj[i]
        if len(neigh) != 1:
            continue
        if syms[neigh[0]] == "O":
            oh.append(i)
    return oh


def heavy_atom_count(atoms: Atoms) -> int:
    return sum(1 for s in atoms.get_chemical_symbols() if s not in ("H", "X"))


def infer_anchor_atom_of_h(parent: Atoms, h_idx: int, covalent_radius_factor: float = 1.25) -> Optional[int]:
    if parent[h_idx].symbol != "H":
        return None

    syms = parent.get_chemical_symbols()
    adj = _bond_graph_by_radii(parent, covalent_radius_factor=covalent_radius_factor)
    neigh = adj[h_idx]
    heavy = [j for j in neigh if syms[j] != "H"]
    if len(heavy) == 1:
        return int(heavy[0])
    if len(heavy) > 1:
        pos = parent.get_positions()
        d = [(j, float(np.linalg.norm(pos[j] - pos[h_idx]))) for j in heavy]
        d.sort(key=lambda x: x[1])
        return int(d[0][0])

    pos = parent.get_positions()
    cand = [i for i, s in enumerate(syms) if s != "H"]
    if not cand:
        return None
    d = [(i, float(np.linalg.norm(pos[i] - pos[h_idx]))) for i in cand]
    d.sort(key=lambda x: x[1])
    return int(d[0][0])


# ============================================================
# 6) Config for Rigid Body Alignment
# ============================================================
@dataclass(frozen=True)
class SubstituteConfig:
    max_local_tries: int = 25
    dihedral_step: int = 10
    stochastic_dihedral: bool = True


@dataclass(frozen=True)
class ExhaustiveEnumConfig:
    # 每个状态最多扫描多少个候选 H 位点；None = 全量
    max_site_candidates_per_state: Optional[int] = None

    # 每一层最终最多保留多少个唯一结构；None = 全量
    max_states_per_depth: Optional[int] = None

    # 整个 parent+group 枚举流程最多尝试多少次 substitute；None = 不限
    max_total_substitution_attempts: Optional[int] = None


# ============================================================
# 7) Rigid Body 拼接算法
# ============================================================
def combine_with_rigid_body(
    parent: Atoms,
    frag: FragEntry,
    dummy1_idx: int,
    filter_cfg: FilterConfig,
    sub_cfg: SubstituteConfig,
    rng: random.Random
) -> Tuple[Optional[Atoms], dict]:
    info = {
        "frag_attach_idx": int(frag.attach_idx),
        "frag_dummy_idx": int(frag.dummy2_idx),
        "frag_attach_symbol": str(frag.atoms[frag.attach_idx].symbol),
    }

    base = parent.copy()
    sub = frag.atoms.copy()

    base_anchor_idx = infer_anchor_atom_of_h(base, dummy1_idx)
    if base_anchor_idx is None:
        return None, {**info, "fail_reason": "NO_ANCHOR_FOR_H"}

    v_base = base.positions[dummy1_idx] - base.positions[base_anchor_idx]
    v_base_norm = v_base / np.linalg.norm(v_base)

    sub_dummy_idx = frag.dummy2_idx
    sub_anchor_idx = frag.attach_idx
    v_sub = sub.positions[sub_dummy_idx] - sub.positions[sub_anchor_idx]
    v_sub_norm = v_sub / np.linalg.norm(v_sub)

    vec1 = v_sub_norm
    vec2 = -v_base_norm
    if not np.allclose(vec1, vec2):
        if np.allclose(vec1, -vec2):
            axis = np.cross(vec1, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(vec1, np.array([0.0, 1.0, 0.0]))
            axis /= np.linalg.norm(axis)
            sub.rotate(180, axis, center=sub.positions[sub_anchor_idx])
        else:
            sub.rotate(vec1, vec2, center=sub.positions[sub_anchor_idx])

    r1 = covalent_radii[base.numbers[base_anchor_idx]]
    r2 = covalent_radii[sub.numbers[sub_anchor_idx]]
    ideal_bond_length = r1 + r2

    sub.positions -= sub.positions[sub_anchor_idx]
    target_pos = base.positions[base_anchor_idx] + v_base_norm * ideal_bond_length
    sub.positions += target_pos

    del sub[sub_dummy_idx]
    del base[dummy1_idx]

    if dummy1_idx < base_anchor_idx:
        base_anchor_idx -= 1
    if sub_dummy_idx < sub_anchor_idx:
        sub_anchor_idx -= 1

    valid_angles = []
    best_angle = 0
    max_min_dist = -1.0
    best_positions = sub.positions.copy()

    for angle in range(0, 360, sub_cfg.dihedral_step):
        test_sub = sub.copy()
        test_sub.rotate(angle, v_base_norm, center=target_pos)

        dist_mat = _pairwise_distances_two(base.positions, test_sub.positions)
        dist_mat[base_anchor_idx, sub_anchor_idx] = 1e9

        min_nonbonded = np.min(dist_mat)

        if min_nonbonded >= filter_cfg.min_distance_clash:
            valid_angles.append((angle, min_nonbonded, test_sub.positions.copy()))

        if min_nonbonded > max_min_dist:
            max_min_dist = min_nonbonded
            best_angle = angle
            best_positions = test_sub.positions.copy()

    if valid_angles and sub_cfg.stochastic_dihedral:
        chosen = rng.choice(valid_angles)
        best_angle, final_min_dist, best_positions = chosen
    else:
        final_min_dist = max_min_dist

    sub.positions = best_positions
    actual_bond_length = ideal_bond_length

    if final_min_dist < filter_cfg.min_distance_clash:
        resolved = False
        for _push_steps in range(1, 5):
            sub.positions += v_base_norm * 0.1
            actual_bond_length += 0.1

            dist_mat = _pairwise_distances_two(base.positions, sub.positions)
            dist_mat[base_anchor_idx, sub_anchor_idx] = 1e9
            current_min = np.min(dist_mat)

            if current_min >= filter_cfg.min_distance_clash:
                final_min_dist = current_min
                resolved = True
                break

        if not resolved:
            return None, {
                **info,
                "fail_reason": "STILL_CLASHING_AFTER_PUSH",
                "best_dist": float(final_min_dist),
            }

    combined = base.copy()
    combined.extend(sub)

    ok, reason = topo_geometry_filter(combined, filter_cfg)
    if not ok:
        return None, {
            **info,
            "fail_reason": f"FILTER_FAIL: {reason}",
            "best_dist": float(final_min_dist),
        }

    info["final"] = {
        "bond_length_ideal": float(ideal_bond_length),
        "bond_length_actual": float(actual_bond_length),
        "dihedral_angle": best_angle,
        "min_nonbonded_dist": float(final_min_dist),
    }
    return combined, info


# ============================================================
# 8) Single substitution
# ============================================================
def substitute_once_rigid(
    mol: Atoms,
    h_idx: int,
    group_name: str,
    frag_lib: Dict[str, FragEntry],
    filter_cfg: FilterConfig,
    sub_cfg: SubstituteConfig,
    rng: random.Random
) -> Tuple[Optional[Atoms], dict]:
    if h_idx < 0 or h_idx >= len(mol):
        return None, {"ok": False, "reason": "BAD_H_IDX"}
    if mol[h_idx].symbol != "H":
        return None, {"ok": False, "reason": "TARGET_NOT_H"}
    if group_name not in frag_lib:
        return None, {"ok": False, "reason": "BAD_GROUP"}

    frag = frag_lib[group_name]
    new_mol, rigid_info = combine_with_rigid_body(
        parent=mol,
        frag=frag,
        dummy1_idx=h_idx,
        filter_cfg=filter_cfg,
        sub_cfg=sub_cfg,
        rng=rng
    )

    if new_mol is None:
        return None, {
            "ok": False,
            "reason": rigid_info.get("fail_reason", "RIGID_FAILED"),
            "replace_h": int(h_idx),
            "group": group_name,
            **rigid_info,
        }

    meta = {
        "ok": True,
        "replace_h": int(h_idx),
        "group": group_name,
        **rigid_info,
    }
    return new_mol, meta


# ============================================================
# 9) Random functionalize
# ============================================================
SAFE_GROUPS_FOR_OH_REMOVAL = ("CH3", "CF3", "CN", "F", "SO2F", "SO2CH3", "COOCH3")


def random_functionalize(
    parent_atoms: Atoms,
    frag_lib: Dict[str, FragEntry],
    rng: random.Random,
    n_steps: int,
    max_heavy: int,
    filter_cfg: FilterConfig,
    sub_cfg: SubstituteConfig,
) -> Tuple[Atoms, List[dict]]:
    mol = parent_atoms.copy()
    steps: List[dict] = []

    def _phase_candidates(phase: str) -> Tuple[List[int], List[str]]:
        if phase == "remove_OH":
            h_list = find_oh_h_indices(mol)
            g_list = [g for g in SAFE_GROUPS_FOR_OH_REMOVAL if g in frag_lib]
            return h_list, g_list
        return list_all_h_indices(mol), list(frag_lib.keys())

    def _try_one_step(phase: str) -> bool:
        nonlocal mol, steps
        for _ in range(sub_cfg.max_local_tries):
            if heavy_atom_count(mol) >= max_heavy:
                return False
            h_list, g_list = _phase_candidates(phase)
            if not h_list or not g_list:
                return False

            h_idx = rng.choice(h_list)
            gname = rng.choice(g_list)

            new_mol, meta = substitute_once_rigid(
                mol=mol,
                h_idx=h_idx,
                group_name=gname,
                frag_lib=frag_lib,
                filter_cfg=filter_cfg,
                sub_cfg=sub_cfg,
                rng=rng
            )

            if new_mol is None:
                continue

            meta["phase"] = phase
            steps.append(meta)
            mol = new_mol
            return True
        return False

    if filter_cfg.forbid_oh_bond:
        while len(steps) < n_steps and heavy_atom_count(mol) < max_heavy:
            if not find_oh_h_indices(mol):
                break
            if not _try_one_step("remove_OH"):
                break

    while len(steps) < n_steps and heavy_atom_count(mol) < max_heavy:
        if not list_all_h_indices(mol):
            break
        if not _try_one_step("random"):
            break

    return mol, steps


# ============================================================
# 10) Fixed-group exact-depth 分层穷举
# ============================================================
def candidate_h_indices_for_fixed_group(
    mol: Atoms,
    group_name: str,
    filter_cfg: FilterConfig,
) -> List[int]:
    """
    固定官能团时的候选位点规则：
    1) 如果 forbid_oh_bond=True 且当前还有 OH-H，则优先只处理 OH-H；
    2) 且只有 SAFE_GROUPS_FOR_OH_REMOVAL 里的 group 才允许替换 OH-H；
    3) 否则返回当前分子全部 H。
    """
    if filter_cfg.forbid_oh_bond:
        oh_h = sorted(find_oh_h_indices(mol))
        if oh_h:
            if group_name not in SAFE_GROUPS_FOR_OH_REMOVAL:
                return []
            return oh_h

    return sorted(list_all_h_indices(mol))


def enumerate_fixed_group_layers(
    parent_atoms: Atoms,
    group_name: str,
    frag_lib: Dict[str, FragEntry],
    rng: random.Random,
    max_depth: int,
    max_heavy: int,
    filter_cfg: FilterConfig,
    sub_cfg: SubstituteConfig,
    enum_cfg: ExhaustiveEnumConfig,
) -> Dict[int, List[dict]]:
    """
    固定一个 group_name，对 depth=1..max_depth 做 exact-depth 分层穷举。

    返回:
        layers[1] = 所有恰好 1 次取代的唯一结构
        layers[2] = 所有恰好 2 次取代的唯一结构
        ...
    特点：
    - 允许在前一步新接上的基团上继续取代
    - 每层按 InChI 去重
    - 不同 depth 分开统计
    """
    if max_depth <= 0:
        return {}
    if group_name not in frag_lib:
        return {}

    layers: Dict[int, List[dict]] = {}
    current_layer = [{
        "atoms": parent_atoms.copy(),
        "steps": [],
    }]

    total_attempts = 0
    hard_stop = False

    for depth in range(1, max_depth + 1):
        next_map = {}

        for state in current_layer:
            mol = state["atoms"]
            prev_steps = state["steps"]

            if heavy_atom_count(mol) >= max_heavy:
                continue

            h_list = candidate_h_indices_for_fixed_group(
                mol=mol,
                group_name=group_name,
                filter_cfg=filter_cfg,
            )

            if enum_cfg.max_site_candidates_per_state is not None:
                h_list = h_list[: int(enum_cfg.max_site_candidates_per_state)]

            if not h_list:
                continue

            for h_idx in h_list:
                if (
                    enum_cfg.max_total_substitution_attempts is not None
                    and total_attempts >= int(enum_cfg.max_total_substitution_attempts)
                ):
                    hard_stop = True
                    break

                total_attempts += 1

                new_mol, meta = substitute_once_rigid(
                    mol=mol,
                    h_idx=h_idx,
                    group_name=group_name,
                    frag_lib=frag_lib,
                    filter_cfg=filter_cfg,
                    sub_cfg=sub_cfg,
                    rng=rng,
                )

                if new_mol is None:
                    continue

                if heavy_atom_count(new_mol) > max_heavy:
                    continue

                passed, smiles, inchi, _ = clean_filter_atoms(new_mol, filter_cfg)
                if not passed or smiles is None or inchi is None:
                    continue

                # 仅在同一 depth 内去重
                if inchi in next_map:
                    continue

                meta["phase"] = "fixed_group_exact"
                meta["fixed_group"] = group_name

                rec = {
                    "atoms": new_mol,
                    "steps": prev_steps + [meta],
                    "smiles": smiles,
                    "inchi": inchi,
                    "depth": depth,
                    "group_name": group_name,
                }
                next_map[inchi] = rec

            if hard_stop:
                break

        layer_records = [next_map[k] for k in sorted(next_map.keys())]

        if enum_cfg.max_states_per_depth is not None:
            layer_records = layer_records[: int(enum_cfg.max_states_per_depth)]

        layers[depth] = layer_records

        if hard_stop or not layer_records:
            break

        current_layer = [{
            "atoms": rec["atoms"],
            "steps": rec["steps"],
        } for rec in layer_records]

    return layers


# ============================================================
# 11) Parallel DB generation (保留旧接口)
# ============================================================
G_PARENTS = None
G_PARENT_IDS = None
G_FRAG_LIB = None
G_MAX_HEAVY = None
G_SEED0 = None
G_FILTER_CFG = None
G_SUB_CFG = None


def _init_worker(parents, parent_ids, frag_lib, max_heavy, seed0, filter_cfg_dict, sub_cfg_dict):
    global G_PARENTS, G_PARENT_IDS, G_FRAG_LIB, G_MAX_HEAVY, G_SEED0, G_FILTER_CFG, G_SUB_CFG
    G_PARENTS = parents
    G_PARENT_IDS = parent_ids
    G_FRAG_LIB = frag_lib
    G_MAX_HEAVY = max_heavy
    G_SEED0 = seed0
    G_FILTER_CFG = FilterConfig(**filter_cfg_dict)
    G_SUB_CFG = SubstituteConfig(
        max_local_tries=int(sub_cfg_dict.get("max_local_tries", 25)),
        dihedral_step=int(sub_cfg_dict.get("dihedral_step", 10)),
        stochastic_dihedral=bool(sub_cfg_dict.get("stochastic_dihedral", True))
    )


def _process_single_attempt(attempt_idx: int):
    try:
        rng = random.Random(G_SEED0 + attempt_idx)
        j = rng.randrange(len(G_PARENTS))
        pid = int(G_PARENT_IDS[j])
        patoms = G_PARENTS[j]

        n_steps = rng.choices([1, 2, 3], weights=[0.70, 0.25, 0.05], k=1)[0]
        new_atoms, steps = random_functionalize(
            parent_atoms=patoms,
            frag_lib=G_FRAG_LIB,
            rng=rng,
            n_steps=int(n_steps),
            max_heavy=G_MAX_HEAVY,
            filter_cfg=G_FILTER_CFG,
            sub_cfg=G_SUB_CFG,
        )
        if not steps:
            return None, None

        passed, smiles, inchi, _ = clean_filter_atoms(new_atoms, G_FILTER_CFG)
        if not passed:
            return None, None

        kv = {
            "source": "fg_random",
            "parent_id": pid,
            "n_steps": int(len(steps)),
            "steps": json.dumps(steps, ensure_ascii=False),
            "smiles": smiles,
            "inchi": inchi,
        }
        return new_atoms, kv
    except Exception:
        return None, None


def build_functionalized_library_db(
    src_db: str,
    dst_db: str,
    target_attempts: int = 150_000,
    seed: int = 0,
    max_heavy: int = 12,
    n_cores: Optional[int] = None,
    write_base: bool = True,
    filter_cfg: Optional[FilterConfig] = None,
    frag_lib: Optional[Dict[str, FragEntry]] = None,
    sub_cfg: Optional[SubstituteConfig] = None,
    debug: bool = True,
):
    if filter_cfg is None:
        filter_cfg = FilterConfig()
    if frag_lib is None:
        frag_lib = make_frag_library_default(seed=seed)
    if sub_cfg is None:
        sub_cfg = SubstituteConfig()
    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)

    src = connect(src_db)
    rows = list(src.select())
    parents = [row.toatoms() for row in rows]
    parent_ids = [row.id for row in rows]

    if debug:
        print("[INFO] build_functionalized_library_db")
        print(f"  src_db = {src_db}")
        print(f"  dst_db = {dst_db}")
        print(f"  target_attempts = {target_attempts}")
        print(f"  parents = {len(parents)}")
        print(f"  n_cores = {n_cores}")
        print(f"  filter_cfg = {filter_cfg}")
        print(f"  sub_cfg = {sub_cfg}")

    inchi_seen = set()
    base_written = 0

    with connect(dst_db) as dst:
        if write_base:
            for pid, patoms in zip(parent_ids, parents):
                passed, smiles, inchi, _ = clean_filter_atoms(patoms, filter_cfg)
                if not passed:
                    continue
                if inchi in inchi_seen:
                    continue
                inchi_seen.add(inchi)
                dst.write(patoms, source="base", parent_id=int(pid), smiles=smiles, inchi=inchi)
                base_written += 1

        remaining = max(0, target_attempts - (len(parents) if write_base else 0))
        tasks = range(remaining)

        success_new = 0
        fail = 0

        initargs = (
            parents,
            parent_ids,
            frag_lib,
            max_heavy,
            seed * 10_000_000,
            filter_cfg.__dict__,
            sub_cfg.__dict__
        )

        with Pool(processes=n_cores, initializer=_init_worker, initargs=initargs) as pool:
            it = tqdm(
                pool.imap_unordered(_process_single_attempt, tasks),
                total=remaining,
                desc="parallel functionalization"
            )
            for atoms, kv in it:
                if atoms is None:
                    fail += 1
                    continue
                inchi = kv["inchi"]
                if inchi in inchi_seen:
                    continue
                inchi_seen.add(inchi)
                dst.write(atoms, key_value_pairs=kv)
                success_new += 1

    return {
        "dst_db": dst_db,
        "n_parents": len(parents),
        "write_base": write_base,
        "base_written_unique": base_written,
        "attempts_total": target_attempts,
        "attempts_remaining": remaining,
        "new_written_unique": success_new,
        "fail_or_filtered": fail,
        "unique_total_written": len(inchi_seen),
        "n_cores": n_cores,
        "filter_cfg": filter_cfg.__dict__,
        "sub_cfg": sub_cfg.__dict__,
    }