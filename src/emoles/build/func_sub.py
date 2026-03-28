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
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import WrapLogs

from emoles.build.CombineMols3D import combine_2_mols_with_dummy


# ============================================================
# 0) RDKit 静默上下文（吸收 warning 到内存）
# ============================================================
@contextlib.contextmanager
def rdkit_silent():
    WrapLogs()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield


# ============================================================
# 1) 纯内存：ASE Atoms -> (smiles, inchi)
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

    with rdkit_silent():
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


# ============================================================
# 2) “干净筛选器”：几何/连通性 + SMILES 规则
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
    forbid_oh_bond: bool = True  # 关键：不允许 O–H bond


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((d * d).sum(-1))
    return dist


def topo_geometry_filter(atoms: Atoms, cfg: FilterConfig) -> Tuple[bool, str]:
    n = len(atoms)
    if n < 2:
        return False, "NOT_MOLECULE"

    pos = atoms.get_positions()
    dist = _pairwise_distances(pos)
    dist = dist + np.eye(n) * 1e9

    if float(dist.min()) < cfg.min_distance_clash:
        return False, "CLASH"

    nums = atoms.get_atomic_numbers()
    r = covalent_radii[nums]
    thresh = (r[:, None] + r[None, :]) * cfg.covalent_radius_factor
    bonded = dist <= thresh
    adj = [list(np.where(bonded[i])[0]) for i in range(n)]

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
# 3) 官能团库（dummy = X）
#    注意：不包含 OH（否则必然产生 O–H bond，会被过滤）
# ============================================================
def frag_F():
    return Atoms("FX", positions=[[0, 0, 0], [1.3, 0, 0]]), 1


def frag_CH3():
    mol = Atoms("CHHHX",
                positions=[
                    [0.000, 0.000, 0.000],
                    [0.000, 0.000, 1.090],
                    [1.026, 0.000, -0.363],
                    [-0.513, 0.889, -0.363],
                    [-0.513, -0.889, -0.363],  # X
                ])
    return mol, 4


def frag_CF3():
    mol = Atoms("CFFFX",
                positions=[
                    [0.000, 0.000, 0.000],
                    [1.330, 0.000, 0.000],
                    [-0.665, 1.152, 0.000],
                    [-0.665, -1.152, 0.000],
                    [0.000, 0.000, 1.090],  # X
                ])
    return mol, 4


def frag_CN():
    mol = Atoms("CNX",
                positions=[
                    [0.000, 0.000, 0.000],   # C (attach)
                    [1.160, 0.000, 0.000],   # N
                    [-1.090, 0.000, 0.000],  # X
                ])
    return mol, 2


def frag_SO2F():
    mol = Atoms("SOOFX",
                positions=[
                    [0.000, 0.000, 0.000],    # S (attach)
                    [1.430, 0.000, 0.000],    # O
                    [-1.430, 0.000, 0.000],   # O
                    [0.000, 1.600, 0.000],    # F
                    [0.000, -1.800, 0.000],   # X
                ])
    return mol, 4


def frag_OCH3():
    # methanol: CH3-O-H，把羟基 H 替换为 X -> -OCH3（O 作为连接点）
    mol = Atoms("COHHHX",
                positions=[
                    [0.000, 0.000, 0.000],   # C
                    [1.430, 0.000, 0.000],   # O
                    [-0.630, 0.910, 0.000],  # H
                    [-0.630, -0.910, 0.000], # H
                    [0.000, 0.000, 1.090],   # H
                    [2.060, 0.000, 0.960],   # X (was OH hydrogen)
                ])
    return mol, 5


def frag_CHO():
    # formaldehyde: H2C=O，把一个 H 替换为 X -> -CHO（C 作为连接点）
    mol = Atoms("COHXH",
                positions=[
                    [0.000, 0.000, 0.000],   # C (attach)
                    [1.210, 0.000, 0.000],   # O
                    [-0.630, 0.910, 0.000],  # H
                    [-0.630, -0.910, 0.000], # X
                    [0.000, 0.000, 1.090],   # H
                ])
    return mol, 3


def frag_COCH3():
    # acetyl: CH3-CO-H，把醛 H 替换为 X -> -COCH3（羰基 C 连接）
    mol = Atoms("CCOHHHHX",
                positions=[
                    [0.000, 0.000, 0.000],    # C (methyl)
                    [1.520, 0.000, 0.000],    # C (carbonyl, attach)
                    [2.730, 0.000, 0.000],    # O
                    [-0.630, 0.910, 0.000],   # H
                    [-0.630, -0.910, 0.000],  # H
                    [0.000, 0.000, 1.090],    # H
                    [1.520, 0.000, 1.090],    # H
                    [1.520, 0.000, -1.090],   # X
                ])
    return mol, 7


def frag_COOCH3():
    # methyl ester: HCOOCH3，把 formyl H 替换为 X -> -COOCH3（羰基 C 连接）
    mol = Atoms("COOCHHHX",
                positions=[
                    [0.000, 0.000, 0.000],    # C (attach)
                    [1.210, 0.000, 0.000],    # O
                    [-1.330, 0.000, 0.000],   # O
                    [-2.760, 0.000, 0.000],   # C
                    [-3.390, 0.910, 0.000],   # H
                    [-3.390, -0.910, 0.000],  # H
                    [-2.760, 0.000, 1.090],   # H
                    [0.000, 0.000, 1.090],    # X
                ])
    return mol, 7


def make_frag_library_default() -> Dict[str, Tuple[Atoms, int, dict]]:
    lib = {}
    for name, builder in [
        ("F", frag_F),
        ("CH3", frag_CH3),
        ("CF3", frag_CF3),
        ("CN", frag_CN),
        ("SO2F", frag_SO2F),
        ("OCH3", frag_OCH3),
        ("CHO", frag_CHO),
        ("COCH3", frag_COCH3),
        ("COOCH3", frag_COOCH3),
    ]:
        frag, d2 = builder()
        lib[name] = (frag, d2, {"skin": -0.5})
    return lib


# ============================================================
# 4) 随机取代：选 H 位点（包括 O–H / 甲基 H）
# ============================================================
def heavy_atom_count(atoms: Atoms) -> int:
    return sum(1 for s in atoms.get_chemical_symbols() if s not in ("H", "X"))


def h_sites_by_anchor(atoms: Atoms, covalent_radius_factor: float = 1.15) -> Dict[str, List[int]]:
    n = len(atoms)
    syms = atoms.get_chemical_symbols()
    nums = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    dist = _pairwise_distances(pos) + np.eye(n) * 1e9

    r = covalent_radii[nums]
    thresh = (r[:, None] + r[None, :]) * covalent_radius_factor
    bonded = dist <= thresh

    buckets: Dict[str, List[int]] = {}
    for i, s in enumerate(syms):
        if s != "H":
            continue
        neigh = list(np.where(bonded[i])[0])
        if len(neigh) != 1:
            continue
        anchor = neigh[0]
        anchor_sym = syms[anchor]
        buckets.setdefault(anchor_sym, []).append(i)
    return buckets


# 注意：不包含 "OH"
ALLOWED_GROUPS_BY_ANCHOR = {
    "C": {"CH3", "OCH3", "CHO", "COCH3", "COOCH3", "F", "CF3", "CN", "SO2F"},
    "O": {"CH3", "OCH3", "CHO", "COCH3", "COOCH3", "F", "CF3", "CN", "SO2F"},
    "N": {"CH3", "OCH3", "CHO", "COCH3", "COOCH3", "F", "CF3", "CN", "SO2F"},
    "S": {"CH3", "OCH3", "CHO", "COCH3", "COOCH3", "F", "CF3", "CN", "SO2F"},
}


def random_functionalize(parent_atoms: Atoms,
                         frag_lib: Dict[str, Tuple[Atoms, int, dict]],
                         rng: random.Random,
                         n_steps: int,
                         max_heavy: int) -> Tuple[Atoms, List[dict]]:
    mol = parent_atoms.copy()
    steps: List[dict] = []

    for _ in range(n_steps):
        if heavy_atom_count(mol) >= max_heavy:
            break

        buckets = h_sites_by_anchor(mol)
        anchors = [a for a in ("C", "O", "N", "S") if a in buckets and len(buckets[a]) > 0]
        if not anchors:
            break

        weights = []
        for a in anchors:
            weights.append(0.65 if a == "C" else 0.35 / max(1, (len(anchors) - (1 if "C" in anchors else 0))))
        anchor = rng.choices(anchors, weights=weights, k=1)[0]
        h_idx = rng.choice(buckets[anchor])

        allowed = list(ALLOWED_GROUPS_BY_ANCHOR.get(anchor, set(frag_lib.keys())))
        gname = rng.choice(allowed)

        frag, dummy2_idx, kwargs = frag_lib[gname]
        mol = combine_2_mols_with_dummy(
            mol1=mol,
            mol2=frag,
            dummy1_idx=h_idx,
            dummy2_idx=dummy2_idx,
            **kwargs
        )
        steps.append({"anchor": anchor, "h_idx": int(h_idx), "group": gname})

    return mol, steps


# ============================================================
# 5) 并行：worker + 对外函数（无 main）
# ============================================================
G_PARENTS = None
G_PARENT_IDS = None
G_FRAG_LIB = None
G_MAX_HEAVY = None
G_SEED0 = None
G_FILTER_CFG = None


def _init_worker(parents, parent_ids, frag_lib, max_heavy, seed0, filter_cfg_dict):
    global G_PARENTS, G_PARENT_IDS, G_FRAG_LIB, G_MAX_HEAVY, G_SEED0, G_FILTER_CFG
    G_PARENTS = parents
    G_PARENT_IDS = parent_ids
    G_FRAG_LIB = frag_lib
    G_MAX_HEAVY = max_heavy
    G_SEED0 = seed0
    G_FILTER_CFG = FilterConfig(**filter_cfg_dict)


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
            n_steps=n_steps,
            max_heavy=G_MAX_HEAVY
        )
        if len(steps) == 0:
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
    frag_lib: Optional[Dict[str, Tuple[Atoms, int, dict]]] = None,
):
    if filter_cfg is None:
        filter_cfg = FilterConfig()
    if frag_lib is None:
        frag_lib = make_frag_library_default()

    src = connect(src_db)
    parents = []
    parent_ids = []
    for row in src.select():
        parents.append(row.toatoms())
        parent_ids.append(row.id)

    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)

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
        )

        with Pool(processes=n_cores, initializer=_init_worker, initargs=initargs) as pool:
            it = tqdm(pool.imap_unordered(_process_single_attempt, tasks),
                      total=remaining,
                      desc="parallel functionalization")
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
    }


# ============================================================
# 6) 方便测试：对单个 Atoms 生成 N 个取代产物（不写 DB）
# ============================================================
def generate_random_substitutions(
    base_atoms: Atoms,
    n: int = 10,
    seed: int = 0,
    n_steps_choices=(1, 2, 3),
    n_steps_weights=(0.7, 0.25, 0.05),
    max_heavy: int = 12,
    filter_cfg: Optional[FilterConfig] = None,
    frag_lib: Optional[Dict[str, Tuple[Atoms, int, dict]]] = None,
) -> List[Tuple[Atoms, dict]]:
    if filter_cfg is None:
        filter_cfg = FilterConfig()
    if frag_lib is None:
        frag_lib = make_frag_library_default()

    rng = random.Random(seed)
    outs: List[Tuple[Atoms, dict]] = []

    max_try = n * 30
    for _ in range(max_try):
        if len(outs) >= n:
            break
        n_steps = rng.choices(list(n_steps_choices), weights=list(n_steps_weights), k=1)[0]
        new_atoms, steps = random_functionalize(
            parent_atoms=base_atoms,
            frag_lib=frag_lib,
            rng=rng,
            n_steps=int(n_steps),
            max_heavy=max_heavy
        )
        if not steps:
            continue

        passed, smiles, inchi, reason = clean_filter_atoms(new_atoms, filter_cfg)
        if not passed:
            continue

        meta = {
            "n_steps": len(steps),
            "steps": steps,
            "smiles": smiles,
            "inchi": inchi,
            "reason": reason,
        }
        outs.append((new_atoms, meta))

    return outs