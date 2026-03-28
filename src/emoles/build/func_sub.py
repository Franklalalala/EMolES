# Copyright AISI
# emoles.build.func_sub

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

from emoles.build.CombineMols3D import combine_2_mols_with_dummy


# ============================================================
# 0) Dynamic distance (基于被取代 H 的位置)
# ============================================================
@dataclass(frozen=True)
class DynamicDistanceConfig:
    # offset in Å; bond_distance = d(anchor-H) + offset
    offset_start: float = 0.0
    step: float = 0.1

    # 同一 offset 下，尝试多少次随机旋转
    rotations_per_offset: int = 6

    # offset 调整次数上限
    max_adjust: int = 12

    # offset 限制，防止跑飞
    min_offset: float = -1.0
    max_offset: float = 1.5

    # 将 (r1+r2) 乘以一个系数再用于 skin 计算（一般保持 1.0）
    radii_scale: float = 1.0


DEFAULT_DIST_CFG = DynamicDistanceConfig()


# ============================================================
# 1) RDKit 日志捕获/静默
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
# 2) ASE Atoms -> smiles/inchi (MolFromXYZBlock + DetermineBonds)
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
    """
    返回 smiles/inchi + RDKit stderr log，用于过滤“奇怪价态/重排”警告。
    """
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
# 3) 筛选器：几何/连通性 + SMILES 规则（禁 O–H）
# ============================================================
@dataclass(frozen=True)
class FilterConfig:
    # geometry/topology
    covalent_radius_factor: float = 1.10
    min_distance_clash: float = 0.55
    require_single_component: bool = True
    forbid_isolated_atom: bool = True

    # smiles rules
    forbid_3_4_member_rings: bool = True
    forbid_cc_triple: bool = True
    forbid_oo_bond: bool = True
    forbid_oh_bond: bool = True

    # 过滤 RDKit 警告（避免怪价态/电荷重排）
    reject_rdkit_warnings: bool = True


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    d = pos[:, None, :] - pos[None, :, :]
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
# 4) 片段库：用 RDKit 可靠 SMILES -> 3D -> 优化 -> 替换 placeholder 为 X
# ============================================================
@dataclass(frozen=True)
class FragEntry:
    atoms: Atoms
    dummy2_idx: int   # X 原子
    attach_idx: int   # 与 X 相连的那个原子（用于半径/距离计算）


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
        # 退一步
        code = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if code != 0:
            raise RuntimeError(f"RDKit Embed failed for: {smiles}")

    # 优先 MMFF，其次 UFF；失败就算了（至少有 3D）
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
        if z == 0:
            syms.append("X")
        else:
            syms.append(pt.GetElementSymbol(z))
        p = conf.GetAtomPosition(a.GetIdx())
        pos.append([float(p.x), float(p.y), float(p.z)])
    return Atoms("".join(syms), positions=np.array(pos, dtype=float))


def frag_from_smiles_by_replacing_terminal(
    smiles: str,
    placeholder_symbol: str = "Cl",
    seed: int = 0,
) -> FragEntry:
    """
    典型做法：用含 Cl 的前体（更稳定、力场可参数化），3D+优化后把 Cl 改成 X。
    要求 placeholder 是 terminal（degree=1）。
    """
    mol = _rdkit_mol_3d(smiles, seed=seed)

    # 找 terminal placeholder
    ph_idx = None
    for a in mol.GetAtoms():
        if a.GetSymbol() == placeholder_symbol and a.GetDegree() == 1:
            ph_idx = a.GetIdx()
            break
    if ph_idx is None:
        raise ValueError(f"no terminal {placeholder_symbol} found in {smiles}")

    attach_idx = mol.GetAtomWithIdx(ph_idx).GetNeighbors()[0].GetIdx()

    # 优化完成后再把 placeholder 改成 dummy 原子（atomic num 0）
    mol.GetAtomWithIdx(ph_idx).SetAtomicNum(0)

    atoms = _rdkit_to_ase_atoms(mol)
    dummy2_idx = int(ph_idx)

    return FragEntry(atoms=atoms, dummy2_idx=dummy2_idx, attach_idx=int(attach_idx))


def make_frag_library_default(seed: int = 0) -> Dict[str, FragEntry]:
    """
    片段来源（SMILES）尽量选“常见真实分子前体”，再替换 terminal Cl/H 为 X。
    - CH3: chloromethane (CCl)
    - CF3: trifluoromethyl chloride (FC(F)(F)Cl)
    - CN : cyanogen chloride (N#CCl) -> X-C#N
    - OCH3: methyl hypochlorite (COCl) -> X-O-CH3
    - COOCH3: methyl chloroformate (COC(=O)Cl) -> X-C(=O)-O-CH3
    - SO2F: sulfonyl fluoride chloride 前体 (O=S(=O)(F)Cl) -> X-S(=O)2F
    - F: hydrogen fluoride ([H]F) -> X-F
    """
    lib: Dict[str, FragEntry] = {}

    # 这些用 Cl 作为 placeholder
    lib["CH3"] = frag_from_smiles_by_replacing_terminal("CCl", "Cl", seed=seed)
    lib["CF3"] = frag_from_smiles_by_replacing_terminal("FC(F)(F)Cl", "Cl", seed=seed)
    lib["CN"] = frag_from_smiles_by_replacing_terminal("N#CCl", "Cl", seed=seed)
    lib["OCH3"] = frag_from_smiles_by_replacing_terminal("COCl", "Cl", seed=seed)
    lib["COOCH3"] = frag_from_smiles_by_replacing_terminal("COC(=O)Cl", "Cl", seed=seed)
    lib["SO2F"] = frag_from_smiles_by_replacing_terminal("O=S(=O)(F)Cl", "Cl", seed=seed)

    # F 用 H 作 placeholder：HF
    lib["F"] = frag_from_smiles_by_replacing_terminal("[H]F", "H", seed=seed)

    return lib


# ============================================================
# 5) 位点识别：H 列表 + O–H 识别
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


# ============================================================
# 6) 随机旋转（用于同一 offset 下尝试构象）
# ============================================================
def _random_rotation_matrix(rng: random.Random) -> np.ndarray:
    # uniform random rotation via quaternion
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    R = np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=float,
    )
    return R


def rotate_atoms_around_index(atoms: Atoms, center_idx: int, rng: random.Random) -> Atoms:
    a = atoms.copy()
    pos = a.get_positions()
    c = pos[center_idx].copy()
    R = _random_rotation_matrix(rng)
    pos2 = (pos - c) @ R.T + c
    a.set_positions(pos2)
    return a


# ============================================================
# 7) 关键 helper：用“被取代 H 的位置”作为默认距离基准
# ============================================================
def infer_anchor_atom_of_h(parent: Atoms, h_idx: int, covalent_radius_factor: float = 1.25) -> Optional[int]:
    """
    找该 H 连接的重原子 anchor。
    - 优先用半径阈值的邻接
    - 不稳时 fallback：最近的非 H 原子
    """
    if parent[h_idx].symbol != "H":
        return None

    syms = parent.get_chemical_symbols()
    adj = _bond_graph_by_radii(parent, covalent_radius_factor=covalent_radius_factor)
    neigh = adj[h_idx]
    heavy = [j for j in neigh if syms[j] != "H"]
    if len(heavy) == 1:
        return int(heavy[0])
    if len(heavy) > 1:
        # 取最近的重原子
        pos = parent.get_positions()
        d = [(j, float(np.linalg.norm(pos[j] - pos[h_idx]))) for j in heavy]
        d.sort(key=lambda x: x[1])
        return int(d[0][0])

    # fallback：全局最近重原子
    pos = parent.get_positions()
    cand = [i for i, s in enumerate(syms) if s != "H"]
    if not cand:
        return None
    d = [(i, float(np.linalg.norm(pos[i] - pos[h_idx]))) for i in cand]
    d.sort(key=lambda x: x[1])
    return int(d[0][0])


def h_based_bond_distance(parent: Atoms, h_idx: int, anchor_idx: int) -> float:
    pos = parent.get_positions()
    return float(np.linalg.norm(pos[anchor_idx] - pos[h_idx]))


def bond_distance_to_skin(
    anchor_z: int,
    attach_z: int,
    bond_distance: float,
    radii_scale: float = 1.0,
) -> float:
    """
    把物理“目标键长”转换成 combine_2_mols_with_dummy 用的 skin。
    假设 combine 内部是按 (r1+r2+skin) 之类控制目标距离。
    """
    r = covalent_radii
    return float(bond_distance - radii_scale * (r[anchor_z] + r[attach_z]))


# ============================================================
# 8) Dynamic distance 拼接：同一 offset 下旋转多次，失败再调 offset
# ============================================================
def _offset_update_from_reasons(reasons: List[str], offset: float, step: float) -> float:
    # clash -> 往大扩；disconnected/isolated -> 往小收
    cnt = {}
    for r in reasons:
        cnt[r] = cnt.get(r, 0) + 1
    n_clash = cnt.get("CLASH", 0)
    n_disc = cnt.get("DISCONNECTED", 0) + cnt.get("ISOLATED_ATOM", 0)

    if n_clash > n_disc:
        return offset + step
    if n_disc > n_clash:
        return offset - step

    # 没明显倾向：不动（交给随机旋转）
    return offset


def combine_with_dynamic_distance(
    parent: Atoms,
    frag: FragEntry,
    dummy1_idx: int,
    rng: random.Random,
    filter_cfg: FilterConfig,
    dist_cfg: DynamicDistanceConfig,
) -> Tuple[Optional[Atoms], dict]:
    """
    默认目标距离：anchor-H 原有位置（d0），用 offset 微调。
    """
    info = {
        "dynamic_distance": {
            "offset_start": dist_cfg.offset_start,
            "step": dist_cfg.step,
            "rotations_per_offset": dist_cfg.rotations_per_offset,
            "max_adjust": dist_cfg.max_adjust,
            "radii_scale": dist_cfg.radii_scale,
            "trials": [],
        }
    }

    anchor_idx = infer_anchor_atom_of_h(parent, dummy1_idx)
    if anchor_idx is None:
        return None, {**info, "fail_reason": "NO_ANCHOR_FOR_H"}

    d0 = h_based_bond_distance(parent, dummy1_idx, anchor_idx)
    anchor_z = int(parent[anchor_idx].number)
    attach_z = int(frag.atoms[frag.attach_idx].number)

    offset = float(dist_cfg.offset_start)

    for k in range(dist_cfg.max_adjust):
        offset = max(dist_cfg.min_offset, min(dist_cfg.max_offset, offset))

        reasons_this_offset: List[str] = []
        for rtry in range(dist_cfg.rotations_per_offset):
            bond_d = d0 + offset
            skin = bond_distance_to_skin(
                anchor_z=anchor_z,
                attach_z=attach_z,
                bond_distance=bond_d,
                radii_scale=dist_cfg.radii_scale,
            )

            frag_rot = rotate_atoms_around_index(frag.atoms, center_idx=frag.dummy2_idx, rng=rng)

            try:
                new_mol = combine_2_mols_with_dummy(
                    mol1=parent.copy(),
                    mol2=frag_rot,
                    dummy1_idx=int(dummy1_idx),
                    dummy2_idx=int(frag.dummy2_idx),
                    skin=float(skin),
                )
            except Exception as e:
                reasons_this_offset.append("COMBINE_EXCEPTION")
                info["dynamic_distance"]["trials"].append(
                    {
                        "adjust_i": k,
                        "rot_i": rtry,
                        "d0": d0,
                        "offset": offset,
                        "bond_d": bond_d,
                        "skin": skin,
                        "reason": "COMBINE_EXCEPTION",
                        "exc": repr(e),
                    }
                )
                continue

            ok, reason = topo_geometry_filter(new_mol, filter_cfg)
            info["dynamic_distance"]["trials"].append(
                {
                    "adjust_i": k,
                    "rot_i": rtry,
                    "d0": d0,
                    "offset": offset,
                    "bond_d": bond_d,
                    "skin": skin,
                    "reason": reason,
                }
            )

            if ok:
                info["dynamic_distance"]["final"] = {
                    "d0": d0,
                    "offset": offset,
                    "bond_d": bond_d,
                    "skin": skin,
                    "adjust_i": k,
                    "rot_i": rtry,
                    "anchor_idx": int(anchor_idx),
                    "attach_idx": int(frag.attach_idx),
                }
                return new_mol, info

            reasons_this_offset.append(reason)

        # 该 offset 下全失败 -> 调 offset
        new_offset = _offset_update_from_reasons(reasons_this_offset, offset=offset, step=dist_cfg.step)
        if new_offset == offset:
            # 没倾向就随机走一步，避免停滞
            new_offset = offset + (dist_cfg.step if rng.random() < 0.5 else -dist_cfg.step)
        offset = new_offset

    return None, {**info, "fail_reason": "DYNAMIC_DISTANCE_FAILED"}


# ============================================================
# 9) 单步替换（基于 dynamic distance）
# ============================================================
@dataclass(frozen=True)
class SubstituteConfig:
    dist_cfg: DynamicDistanceConfig = DEFAULT_DIST_CFG
    max_local_tries: int = 25


def substitute_once_dynamic(
    mol: Atoms,
    h_idx: int,
    group_name: str,
    frag_lib: Dict[str, FragEntry],
    rng: random.Random,
    filter_cfg: FilterConfig,
    sub_cfg: SubstituteConfig,
) -> Tuple[Optional[Atoms], dict]:
    if h_idx < 0 or h_idx >= len(mol):
        return None, {"ok": False, "reason": "BAD_H_IDX"}
    if mol[h_idx].symbol != "H":
        return None, {"ok": False, "reason": "TARGET_NOT_H"}
    if group_name not in frag_lib:
        return None, {"ok": False, "reason": "BAD_GROUP"}

    frag = frag_lib[group_name]
    new_mol, dyn_info = combine_with_dynamic_distance(
        parent=mol,
        frag=frag,
        dummy1_idx=h_idx,
        rng=rng,
        filter_cfg=filter_cfg,
        dist_cfg=sub_cfg.dist_cfg,
    )
    if new_mol is None:
        return None, {
            "ok": False,
            "reason": dyn_info.get("fail_reason", "DYNAMIC_FAILED"),
            "replace_h": int(h_idx),
            "group": group_name,
            **dyn_info,
        }

    meta = {
        "ok": True,
        "replace_h": int(h_idx),
        "group": group_name,
        **dyn_info,
    }
    return new_mol, meta


# ============================================================
# 10) 随机取代：禁 O–H 时先消掉所有 O–H
# ============================================================
SAFE_GROUPS_FOR_OH_REMOVAL = ("CH3", "CF3", "CN", "F", "SO2F", "COOCH3")
# 注意：不要用 OCH3 去替换 O–H，容易引入 O–O（被 forbid_oo_bond 过滤）


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
        else:
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

            new_mol, meta = substitute_once_dynamic(
                mol=mol,
                h_idx=h_idx,
                group_name=gname,
                frag_lib=frag_lib,
                rng=rng,
                filter_cfg=filter_cfg,
                sub_cfg=sub_cfg,
            )
            if new_mol is None:
                continue

            meta["phase"] = phase
            steps.append(meta)
            mol = new_mol
            return True
        return False

    # Phase A: 强制消除 O–H
    if filter_cfg.forbid_oh_bond:
        while len(steps) < n_steps and heavy_atom_count(mol) < max_heavy:
            if not find_oh_h_indices(mol):
                break
            if not _try_one_step("remove_OH"):
                break

    # Phase B: 普通随机取代
    while len(steps) < n_steps and heavy_atom_count(mol) < max_heavy:
        if not list_all_h_indices(mol):
            break
        if not _try_one_step("random"):
            break

    return mol, steps


# ============================================================
# 11) 并行生成数据库
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
    dist_cfg = DynamicDistanceConfig(**sub_cfg_dict["dist_cfg"])
    G_SUB_CFG = SubstituteConfig(dist_cfg=dist_cfg, max_local_tries=int(sub_cfg_dict["max_local_tries"]))


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
            if debug:
                print("[INFO] writing base molecules (unique by InChI) ...")
            for pid, patoms in zip(parent_ids, parents):
                passed, smiles, inchi, _ = clean_filter_atoms(patoms, filter_cfg)
                if not passed:
                    continue
                if inchi in inchi_seen:
                    continue
                inchi_seen.add(inchi)
                dst.write(patoms, source="base", parent_id=int(pid), smiles=smiles, inchi=inchi)
                base_written += 1
            if debug:
                print(f"[INFO] base_written_unique = {base_written}")

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
            {"dist_cfg": sub_cfg.dist_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
        )

        if debug:
            print("[INFO] starting parallel generation ...")
        with Pool(processes=n_cores, initializer=_init_worker, initargs=initargs) as pool:
            it = tqdm(pool.imap_unordered(_process_single_attempt, tasks), total=remaining, desc="parallel functionalization")
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

    if debug:
        print("[INFO] done.")
        print(f"  new_written_unique = {success_new}")
        print(f"  fail_or_filtered = {fail}")
        print(f"  unique_total_written = {len(inchi_seen)}")

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
        "sub_cfg": {"dist_cfg": sub_cfg.dist_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
    }


# ============================================================
# 12) Debug/测试
# ============================================================
def debug_case_like_user_script(base_atoms: Atoms, seed: int = 0):
    """
    复刻你的对照：取第一个 H，替换 CH3 / F，打印过滤结果。
    """
    print("=== DEBUG: case_like_user_script ===")
    print(f"[INFO] base formula = {base_atoms.get_chemical_formula()}")
    h_list = [a.index for a in base_atoms if a.symbol == "H"]
    print(f"[INFO] base H count = {len(h_list)}")
    if not h_list:
        print("[WARN] base has no H")
        return

    target_h_idx = h_list[0]
    print(f"[INFO] target_h_idx = {target_h_idx}")

    frag_lib = make_frag_library_default(seed=seed)
    cfg = FilterConfig()
    sub_cfg = SubstituteConfig(dist_cfg=DEFAULT_DIST_CFG)
    rng = random.Random(seed)

    for g in ["CH3", "F", "CF3", "CN"]:
        new_mol, meta = substitute_once_dynamic(
            mol=base_atoms,
            h_idx=target_h_idx,
            group_name=g,
            frag_lib=frag_lib,
            rng=rng,
            filter_cfg=cfg,
            sub_cfg=sub_cfg,
        )
        if new_mol is None:
            print(f"[INFO] {g} substitute FAILED reason={meta.get('reason')}")
            continue

        passed, smiles, inchi, reason = clean_filter_atoms(new_mol, cfg)
        print(f"[INFO] {g} smiles={smiles} pass={passed} reason={reason}")

        fin = meta.get("dynamic_distance", {}).get("final", None)
        if fin:
            print(f"[INFO] {g} final bond_d={fin.get('bond_d'):.3f} offset={fin.get('offset'):.3f} skin={fin.get('skin'):.3f}")

    print(f"[INFO] base inferred OH-H count = {len(find_oh_h_indices(base_atoms))}")
    print("=== END DEBUG ===")


def generate_random_substitutions(
    base_atoms: Atoms,
    n: int = 10,
    seed: int = 0,
    max_heavy: int = 12,
    filter_cfg: Optional[FilterConfig] = None,
    frag_lib: Optional[Dict[str, FragEntry]] = None,
    sub_cfg: Optional[SubstituteConfig] = None,
    max_try: int = 500,
    debug: bool = True,
    debug_every: int = 25,
) -> List[Tuple[Atoms, dict]]:
    if filter_cfg is None:
        filter_cfg = FilterConfig()
    if frag_lib is None:
        frag_lib = make_frag_library_default(seed=seed)
    if sub_cfg is None:
        sub_cfg = SubstituteConfig()

    rng = random.Random(seed)
    outs: List[Tuple[Atoms, dict]] = []
    fail_counter: Dict[str, int] = {}

    if debug:
        print("=== generate_random_substitutions DEBUG ===")
        print(f"[INFO] requested n = {n}")
        print(f"[INFO] max_try = {max_try}")
        print(f"[INFO] max_heavy = {max_heavy}")
        print(f"[INFO] base formula = {base_atoms.get_chemical_formula()}")
        print(f"[INFO] base natoms = {len(base_atoms)}")
        print(f"[INFO] base H count = {len([a for a in base_atoms if a.symbol=='H'])}")
        print(f"[INFO] base inferred OH-H count = {len(find_oh_h_indices(base_atoms))}")
        print(f"[INFO] filter_cfg = {filter_cfg}")
        print(f"[INFO] sub_cfg = {sub_cfg}")

    for t in range(1, max_try + 1):
        if len(outs) >= n:
            break

        n_steps = rng.choices([1, 2, 3], weights=[0.70, 0.25, 0.05], k=1)[0]
        new_atoms, steps = random_functionalize(
            parent_atoms=base_atoms,
            frag_lib=frag_lib,
            rng=rng,
            n_steps=int(n_steps),
            max_heavy=max_heavy,
            filter_cfg=filter_cfg,
            sub_cfg=sub_cfg,
        )

        if not steps:
            fail_counter["NO_STEPS"] = fail_counter.get("NO_STEPS", 0) + 1
            continue

        passed, smiles, inchi, reason = clean_filter_atoms(new_atoms, filter_cfg)
        if not passed:
            fail_counter[reason] = fail_counter.get(reason, 0) + 1
            if debug and (t % debug_every == 0):
                print(f"[DEBUG] try={t} failed reason={reason} steps={steps[:1]} ...")
            continue

        meta = {
            "n_steps": len(steps),
            "steps": steps,
            "smiles": smiles,
            "inchi": inchi,
            "sub_cfg": {"dist_cfg": sub_cfg.dist_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
        }
        outs.append((new_atoms, meta))

        if debug:
            print(f"[OK] got {len(outs)}/{n} | steps={len(steps)} | smiles={smiles}")

    if debug:
        print("[INFO] done.")
        print(f"[INFO] generated = {len(outs)} / requested = {n}")
        print("[INFO] fail reason counts:")
        for k in sorted(fail_counter.keys()):
            print(f"  {k}: {fail_counter[k]}")
        print("=== END DEBUG ===")

    return outs