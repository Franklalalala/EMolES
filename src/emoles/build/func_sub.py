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
from ase.build import molecule as ase_molecule
from ase.data import covalent_radii
from ase.db import connect
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import WrapLogs

from emoles.build.CombineMols3D import combine_2_mols_with_dummy


# ============================================================
# 0) Dynamic skin 配置（你要求：start=-0.5, step=0.1）
# ============================================================
@dataclass(frozen=True)
class DynamicSkinConfig:
    start: float = -0.5
    step: float = 0.1

    # 在同一 skin 下尝试多少次“随机旋转再拼接”
    rotations_per_skin: int = 5

    # skin 自适应调整次数上限（总共尝试 = rotations_per_skin * max_adjust）
    max_adjust: int = 10

    # 允许的 skin 取值范围，避免无限跑飞
    min_skin: float = -2.0
    max_skin: float = 2.0


DEFAULT_SKIN_CFG = DynamicSkinConfig()


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
# 2) 纯内存：ASE Atoms -> smiles/inchi
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


def atoms_to_smiles_inchi_fast_verbose(atoms: Atoms) -> Tuple[Optional[str], Optional[str], str]:
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
# 3) 干净筛选器：几何/连通性 + SMILES 规则（禁 O–H）
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


def clean_filter_atoms(atoms: Atoms, cfg: FilterConfig) -> Tuple[bool, Optional[str], Optional[str], str]:
    ok, reason = topo_geometry_filter(atoms, cfg)
    if not ok:
        return False, None, None, reason

    smiles, inchi = atoms_to_smiles_inchi_fast(atoms)
    if smiles is None or inchi is None:
        return False, None, None, "RDKit_FAIL"

    ok2, reason2 = smiles_rule_filter(smiles, cfg)
    if not ok2:
        return False, None, None, reason2

    return True, smiles, inchi, "OK"


# ============================================================
# 4) 片段库（dummy = X）
#    注意：frag_lib 不再携带 skin；skin 由 dynamic skin 在 combine 时动态传入
# ============================================================
@dataclass(frozen=True)
class FragEntry:
    atoms: Atoms
    dummy2_idx: int


def frag_CH3() -> FragEntry:
    m = ase_molecule("CH4")
    dummy2_idx = 4
    m.symbols[dummy2_idx] = "X"
    return FragEntry(m, dummy2_idx)


def frag_F() -> FragEntry:
    m = Atoms("FX", positions=[[0, 0, 0], [1.0, 0, 0]])
    return FragEntry(m, 1)


def frag_CF3() -> FragEntry:
    try:
        m = ase_molecule("CHF3")
        h_idx = [a.index for a in m if a.symbol == "H"][0]
        m.symbols[h_idx] = "X"
        return FragEntry(m, h_idx)
    except Exception:
        m = Atoms(
            "CFFFH",
            positions=[
                [0.000, 0.000, 0.000],
                [1.330, 0.000, 0.000],
                [-0.665, 1.152, 0.000],
                [-0.665, -1.152, 0.000],
                [0.000, 0.000, 1.090],
            ],
        )
        dummy2_idx = 4
        m.symbols[dummy2_idx] = "X"
        return FragEntry(m, dummy2_idx)


def frag_CN() -> FragEntry:
    m = ase_molecule("HCN")
    h_idx = [a.index for a in m if a.symbol == "H"][0]
    m.symbols[h_idx] = "X"
    return FragEntry(m, h_idx)


def frag_SO2F() -> FragEntry:
    m = Atoms(
        "SOOFX",
        positions=[
            [0.000, 0.000, 0.000],  # S (attach)
            [1.430, 0.000, 0.000],  # O
            [-1.430, 0.000, 0.000],  # O
            [0.000, 1.600, 0.000],  # F
            [0.000, -1.800, 0.000],  # X
        ],
    )
    return FragEntry(m, 4)


def frag_OCH3() -> FragEntry:
    # 用 CH3OH，把 OH 上的 H 变成 X（片段自身不含 O–H）
    m = ase_molecule("CH3OH")
    o_idx = [a.index for a in m if a.symbol == "O"][0]
    h_indices = [a.index for a in m if a.symbol == "H"]
    pos = m.get_positions()
    dummy2_idx = sorted(
        [(hi, float(np.linalg.norm(pos[hi] - pos[o_idx]))) for hi in h_indices],
        key=lambda x: x[1],
    )[0][0]
    m.symbols[dummy2_idx] = "X"
    return FragEntry(m, dummy2_idx)


def frag_COOCH3() -> FragEntry:
    m = Atoms(
        "COOCHHHH",
        positions=[
            [0.000, 0.000, 0.000],  # C (attach)
            [1.210, 0.000, 0.000],  # O
            [-1.330, 0.000, 0.000],  # O
            [-2.760, 0.000, 0.000],  # C
            [-3.390, 0.910, 0.000],  # H
            [-3.390, -0.910, 0.000],  # H
            [-2.760, 0.000, 1.090],  # H
            [0.000, 0.000, 1.090],  # H(dummy)
        ],
    )
    dummy2_idx = 7
    m.symbols[dummy2_idx] = "X"
    return FragEntry(m, dummy2_idx)


def make_frag_library_default() -> Dict[str, FragEntry]:
    return {
        "CH3": frag_CH3(),
        "F": frag_F(),
        "CF3": frag_CF3(),
        "CN": frag_CN(),
        "SO2F": frag_SO2F(),
        "OCH3": frag_OCH3(),
        "COOCH3": frag_COOCH3(),
    }


# ============================================================
# 5) 位点识别：H 列表 + O–H 识别
# ============================================================
def list_all_h_indices(atoms: Atoms) -> List[int]:
    return [a.index for a in atoms if a.symbol == "H"]


def find_oh_h_indices(atoms: Atoms, covalent_radius_factor: float = 1.15) -> List[int]:
    """
    用 covalent radii 推断 O–H：返回所有“与某个 O 只有一个邻居关系”的 H
    """
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
# 6) 随机旋转工具（用于“同一 skin 下尝试旋转等”）
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

    # rotation matrix
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
# 7) Dynamic skin 拼接：CLASH -> skin 增大；DISCONNECTED/ISOLATED -> skin 减小
# ============================================================
def _skin_update_from_reasons(reasons: List[str], skin: float, step: float) -> float:
    """
    你描述的规则：
      - clash -> 往大了扩（skin += step）
      - disconnected -> 往小了扩（skin -= step）
    """
    if not reasons:
        return skin

    # 统计
    cnt = {}
    for r in reasons:
        cnt[r] = cnt.get(r, 0) + 1

    n_clash = cnt.get("CLASH", 0)
    n_disc = cnt.get("DISCONNECTED", 0) + cnt.get("ISOLATED_ATOM", 0)

    if n_clash > n_disc:
        return skin + step
    if n_disc > n_clash:
        return skin - step

    # 持平：不改（继续靠旋转/随机性）
    return skin


def combine_with_dynamic_skin(
    mol1: Atoms,
    mol2: Atoms,
    dummy1_idx: int,
    dummy2_idx: int,
    rng: random.Random,
    filter_cfg: FilterConfig,
    skin_cfg: DynamicSkinConfig,
) -> Tuple[Optional[Atoms], dict]:
    """
    返回:
      (new_atoms or None, info)
    info 包含每次尝试的 skin / reason 统计，便于 debug。
    """
    info = {
        "dynamic_skin": {
            "start": skin_cfg.start,
            "step": skin_cfg.step,
            "rotations_per_skin": skin_cfg.rotations_per_skin,
            "max_adjust": skin_cfg.max_adjust,
            "trials": [],
        }
    }

    skin = float(skin_cfg.start)
    for skin_try in range(skin_cfg.max_adjust):
        # clamp
        skin = max(skin_cfg.min_skin, min(skin_cfg.max_skin, skin))

        reasons_this_skin: List[str] = []
        for rot_try in range(skin_cfg.rotations_per_skin):
            frag_rot = rotate_atoms_around_index(mol2, center_idx=dummy2_idx, rng=rng)

            try:
                new_mol = combine_2_mols_with_dummy(
                    mol1=mol1.copy(),
                    mol2=frag_rot,
                    dummy1_idx=int(dummy1_idx),
                    dummy2_idx=int(dummy2_idx),
                    skin=float(skin),
                )
            except Exception as e:
                reasons_this_skin.append("COMBINE_EXCEPTION")
                info["dynamic_skin"]["trials"].append(
                    {"skin": skin, "skin_try": skin_try, "rot_try": rot_try, "reason": "COMBINE_EXCEPTION", "exc": repr(e)}
                )
                continue

            ok, reason = topo_geometry_filter(new_mol, filter_cfg)
            info["dynamic_skin"]["trials"].append(
                {"skin": skin, "skin_try": skin_try, "rot_try": rot_try, "reason": reason}
            )
            if ok:
                info["dynamic_skin"]["final_skin"] = skin
                info["dynamic_skin"]["final_skin_try"] = skin_try
                info["dynamic_skin"]["final_rot_try"] = rot_try
                return new_mol, info

            reasons_this_skin.append(reason)

        # 本 skin 全失败 -> 根据失败类型更新 skin
        new_skin = _skin_update_from_reasons(reasons_this_skin, skin=skin, step=skin_cfg.step)
        if new_skin == skin:
            # 没有明确方向时也走一步随机扰动，避免卡死
            new_skin = skin + (skin_cfg.step if rng.random() < 0.5 else -skin_cfg.step)
        skin = new_skin

    return None, info


# ============================================================
# 8) 单步替换：调用 dynamic skin combine
# ============================================================
@dataclass(frozen=True)
class SubstituteConfig:
    skin_cfg: DynamicSkinConfig = DEFAULT_SKIN_CFG
    max_local_tries: int = 20  # 每一步（选 H/选基团/拼接）总尝试上限


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

    frag_entry = frag_lib[group_name]
    new_mol, dyn_info = combine_with_dynamic_skin(
        mol1=mol,
        mol2=frag_entry.atoms,
        dummy1_idx=int(h_idx),
        dummy2_idx=int(frag_entry.dummy2_idx),
        rng=rng,
        filter_cfg=filter_cfg,
        skin_cfg=sub_cfg.skin_cfg,
    )
    if new_mol is None:
        return None, {
            "ok": False,
            "reason": "DYNAMIC_SKIN_FAILED",
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
# 9) 随机取代：禁 O–H 时先消掉所有 O–H
# ============================================================
SAFE_GROUPS_FOR_OH_REMOVAL = ("CH3", "CF3", "CN", "F", "SO2F", "COOCH3")
# 注意：不要用 "OCH3" 去替换 O–H，否则容易形成 O–O（会被 forbid_oo_bond 过滤）


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

    def _try_one_step(h_candidates: List[int], g_candidates: List[str], phase: str) -> bool:
        nonlocal mol, steps
        if not h_candidates or not g_candidates:
            return False

        for _ in range(sub_cfg.max_local_tries):
            if heavy_atom_count(mol) >= max_heavy:
                return False

            # 重新计算候选（因为 mol 在变化）
            if phase == "remove_OH":
                h_candidates2 = find_oh_h_indices(mol)
            else:
                h_candidates2 = list_all_h_indices(mol)

            if not h_candidates2:
                return False

            h_idx = rng.choice(h_candidates2)
            gname = rng.choice(g_candidates)

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
            oh_list = find_oh_h_indices(mol)
            if not oh_list:
                break
            g_candidates = [g for g in SAFE_GROUPS_FOR_OH_REMOVAL if g in frag_lib]
            ok = _try_one_step(oh_list, g_candidates, phase="remove_OH")
            if not ok:
                break

    # Phase B: 普通随机取代
    while len(steps) < n_steps and heavy_atom_count(mol) < max_heavy:
        h_list = list_all_h_indices(mol)
        if not h_list:
            break
        g_candidates = list(frag_lib.keys())
        ok = _try_one_step(h_list, g_candidates, phase="random")
        if not ok:
            break

    return mol, steps


# ============================================================
# 10) 并行：worker + 对外函数
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

    skin_cfg = DynamicSkinConfig(**sub_cfg_dict["skin_cfg"])
    G_SUB_CFG = SubstituteConfig(skin_cfg=skin_cfg, max_local_tries=int(sub_cfg_dict["max_local_tries"]))


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
        frag_lib = make_frag_library_default()
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
            {"skin_cfg": sub_cfg.skin_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
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
        "sub_cfg": {"skin_cfg": sub_cfg.skin_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
    }


# ============================================================
# 11) Debug/测试：复刻你案例 + 单分子生成 N 个取代产物
# ============================================================
def debug_case_like_user_script(base_atoms: Atoms, seed: int = 0):
    print("=== DEBUG: case_like_user_script ===")
    print(f"[INFO] base formula = {base_atoms.get_chemical_formula()}")
    h_list = [a.index for a in base_atoms if a.symbol == "H"]
    print(f"[INFO] base H count = {len(h_list)}")
    if not h_list:
        print("[WARN] base has no H")
        return

    target_h_idx = h_list[0]
    print(f"[INFO] target_h_idx = {target_h_idx}")

    frag_lib = make_frag_library_default()
    cfg = FilterConfig()
    sub_cfg = SubstituteConfig(skin_cfg=DEFAULT_SKIN_CFG)
    rng = random.Random(seed)

    for g in ["CH3", "F"]:
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
            print(f"[INFO] {g} substitute FAILED meta={meta.get('reason')}")
            continue

        smi, inchi, log = atoms_to_smiles_inchi_fast_verbose(new_mol)
        print(f"[INFO] {g} substituted atoms: {new_mol.get_chemical_formula()}")
        print(f"[INFO] {g} smiles:", smi)
        print(f"[INFO] {g} inchi :", inchi)
        passed, _, _, reason = clean_filter_atoms(new_mol, cfg)
        print(f"[INFO] {g} filter_pass:", passed, "reason:", reason)
        if smi is None:
            print("[RDKit LOG]\n", log)

        dyn = meta.get("dynamic_skin", {})
        if dyn:
            print(f"[INFO] {g} dynamic_skin final_skin = {dyn.get('final_skin', None)}")

    oh_list = find_oh_h_indices(base_atoms)
    print(f"[INFO] base OH-H count (geom inferred) = {len(oh_list)}")
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
        frag_lib = make_frag_library_default()
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
            "sub_cfg": {"skin_cfg": sub_cfg.skin_cfg.__dict__, "max_local_tries": sub_cfg.max_local_tries},
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