from typing import Tuple, List, Union, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import BlockLogs
import numpy as np
from ase import Atoms

# -------- Constants --------
SIMPLE_IONS = {'LI': 'Li', 'NA': 'Na', 'K': 'K', 'MG': 'Mg', 'CA': 'Ca', 'ZN': 'Zn'}

# 添加复杂阴离子的识别模式
COMPLEX_ANIONS = {
    'BF4': {'center': 'B', 'ligand': 'F', 'count': 4},  # BF4⁻
    'PF6': {'center': 'P', 'ligand': 'F', 'count': 6},  # PF6⁻
}

HSAB_SCALE = {  # Higher values = harder base
    "O": 1.00, "F": 0.95, "N": 0.80, "Cl": 0.70,
    "S": 0.60, "Br": 0.50, "I": 0.40, "P": 0.40
}

# Base score weights
WEIGHTS = dict(charge=2.0, hsab=0.5, steric=1.0)

# Geometric params (kept for angle model; no hard gate used now)
GEOMETRIC = {"MIN": 1.5, "MAX": 3.25}

# Steric synergy: midpoint crowding radius
STERIC_PARAMS = dict(mid_radius=2.0)

# Pairwise synergy blend (weights for steric vs angle)
SYNERGY_WEIGHTS = dict(steric=0.7, angle=0.3)

# Target bite angle model (degrees)
BITE_ANGLE = dict(ideal_deg=90.0, tol_deg=20.0)

# Approximate M–donor bond lengths (Å) by element type
DONOR_ML_DIST = {
    "O": 2.10, "N": 2.15, "S": 2.30, "P": 2.35,
    "F": 2.00, "Cl": 2.40, "Br": 2.50, "I": 2.60
}
DEFAULT_ML_DIST = 2.20


# -------- Chemistry Conversion Functions --------
def atoms_to_plain_xyz_block(atoms: Atoms) -> str:
    """Convert ASE Atoms to plain XYZ string (symbols + 3D coords only)."""
    symbols = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    lines = [str(len(symbols)), "generated_by_ase"]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    return "\n".join(lines) + "\n"


def atom_2_mol(atoms: Atoms, charge: int = 0) -> Chem.Mol:
    """
    Convert ASE Atoms to RDKit Mol via in-memory XYZ block,
    providing total charge information for accurate bond perception.
    """
    xyz_block = atoms_to_plain_xyz_block(atoms)
    with BlockLogs():
        mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to generate molecule from XYZ data")

    mol = Chem.Mol(mol)  # writable copy
    if charge == 0:
        DetermineBonds(mol, charge=charge, useHueckel=True)
    else:
        DetermineBonds(mol, charge=charge, useHueckel=False)

    return mol


def mol_2_atoms(mol: Chem.Mol) -> Atoms:
    """Convert RDKit Mol to ASE Atoms."""
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    return Atoms(symbols=symbols, positions=positions)


# -------- Helper function for detecting complex anions --------
def _detect_complex_anion(identifier: Union[str, Atoms, Chem.Mol]) -> Optional[Tuple[str, Atoms, int]]:
    """
    检测是否为BF4⁻或PF6⁻阴离子
    返回: (anion_name, ase_atoms, f_index) 或 None
    """
    # 将输入转换为ASE Atoms对象以便分析
    if isinstance(identifier, str):
        # 尝试从SMILES创建分子
        try:
            mol = Chem.MolFromSmiles(identifier)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            ase_atoms = mol_2_atoms(mol)
        except:
            return None
    elif isinstance(identifier, Atoms):
        ase_atoms = identifier
    elif isinstance(identifier, Chem.Mol):
        ase_atoms = mol_2_atoms(identifier)
    else:
        return None

    # 获取化学符号
    symbols = ase_atoms.get_chemical_symbols()

    # 检测每种阴离子模式
    for anion_name, pattern in COMPLEX_ANIONS.items():
        center_symbol = pattern['center']
        ligand_symbol = pattern['ligand']
        expected_count = pattern['count']

        # 统计原子
        center_count = symbols.count(center_symbol)
        ligand_count = symbols.count(ligand_symbol)

        # 检查是否匹配模式（1个中心原子 + 预期数量的配体原子）
        if center_count == 1 and ligand_count == expected_count:
            # 如果只有这些原子（可能还有氢），则认为是该阴离子
            total_non_h = sum(1 for s in symbols if s != 'H')
            if total_non_h == (1 + expected_count):
                # 找到第一个F原子的索引
                for idx, symbol in enumerate(symbols):
                    if symbol == ligand_symbol:
                        return anion_name, ase_atoms, idx

    return None


# -------- Input Processing Functions --------
def _parse_input(identifier: Union[str, Atoms, Chem.Mol], total_charge=None) -> Chem.Mol:
    """Unified input: SMILES, ASE Atoms, or RDKit Mol."""
    if isinstance(identifier, str):
        return _smiles_to_mol(identifier)
    elif isinstance(identifier, Atoms):
        # <<< NEW LOGIC: Estimate total charge from ase.Atoms

        if not total_charge:
            if identifier.has('initial_charges'):
                # Sum of initial charges, rounded to the nearest integer
                charge_sum = identifier.get_initial_charges().sum()
                total_charge = int(round(charge_sum))
            elif identifier.has('charge'):
                total_charge = int(identifier.charge)
            else:
                try:
                    total_charge = int(identifier.charge)
                except:
                    total_charge = 0

        # Pass the estimated charge to the conversion function
        return atom_2_mol(identifier, charge=total_charge)
    elif isinstance(identifier, Chem.Mol):
        # For RDKit Mol, assume charge is already correctly handled or neutral
        # We can also get charge if it's set: Chem.GetFormalCharge(mol)
        return _ensure_3d_mol(identifier)
    else:
        raise TypeError(f"Unsupported input type: {type(identifier)}")


def _smiles_to_mol(smiles: str) -> Chem.Mol:
    """SMILES -> RDKit Mol with 3D coords."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def _ensure_3d_mol(mol: Chem.Mol) -> Chem.Mol:
    """Ensure Mol has 3D coords."""
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    return mol


# -------- Scoring Functions (base score) --------
def _calculate_partial_charges(mol: Chem.Mol) -> np.ndarray:
    """MMFF partial charges; fallback to Gasteiger."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            charges = [mp.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
            return np.array(charges)
        else:
            raise ValueError("MMFF parameters not available, falling back to Gasteiger.")
    except Exception:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                charge = atom.GetDoubleProp('_GasteigerCharge')
                if np.isnan(charge):
                    charge = 0.0
            except Exception:
                charge = 0.0
            charges.append(charge)
        return np.array(charges)


def _compute_atom_scores(mol: Chem.Mol, charges: np.ndarray) -> List[Tuple[int, float]]:
    """Score all heavy atoms."""
    scores = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        score = _calculate_score(atom, charges[i])
        scores.append((i, score))
    return scores


def _calculate_score(atom: Chem.Atom, charge: float) -> float:
    """Weighted score: charge + HSAB + sterics (topology)."""
    charge_score = -charge  # more negative is better
    hsab_score = HSAB_SCALE.get(atom.GetSymbol(), 0.3)
    steric_score = _calculate_steric_score(atom)
    symbol = atom.GetSymbol()
    total_score = (WEIGHTS["charge"] * charge_score +
                   WEIGHTS["hsab"] * hsab_score +
                   WEIGHTS["steric"] * steric_score)

    # 3. 优化打印信息：将所有相关信息格式化后在一行内打印
    #    - {symbol:<2} 表示原子符号占2个字符宽度，左对齐
    #    - {:6.3f} 表示浮点数占6个字符宽度，保留3位小数

    # print(
    #     f"Atom: {symbol:<2} | "
    #     f"Scores -> Charge: {charge_score:6.3f}, HSAB: {hsab_score:6.3f}, Steric: {steric_score:6.3f} | "
    #     f"Total: {total_score:7.3f}"
    # )


    return (WEIGHTS["charge"] * charge_score +
            WEIGHTS["hsab"] * hsab_score +
            WEIGHTS["steric"] * steric_score)


def _calculate_steric_score(atom: Chem.Atom) -> float:
    """Topology sterics: fewer heavy neighbors is better."""
    heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
    return 1.0 / (1 + heavy_neighbors)


# -------- Geometry helpers for synergy --------
def _np_pos(conf: Chem.Conformer, idx: int) -> np.ndarray:
    p = conf.GetAtomPosition(idx)
    return np.array([p.x, p.y, p.z], dtype=float)


def _midpoint_crowding(mol: Chem.Mol, a_idx: int, b_idx: int, radius: float = 2.0) -> float:
    """
    Steric synergy proxy in [0,1]: fewer heavy atoms around the A–B midpoint yields higher value.
    """
    conf = mol.GetConformer()
    pa = _np_pos(conf, a_idx)
    pb = _np_pos(conf, b_idx)
    mid = 0.5 * (pa + pb)
    count = 0
    for j in range(mol.GetNumAtoms()):
        if j == a_idx or j == b_idx:
            continue
        aj = mol.GetAtomWithIdx(j)
        if aj.GetAtomicNum() == 1:
            continue
        pj = _np_pos(conf, j)
        if np.linalg.norm(pj - mid) < radius:
            count += 1
    return float(1.0 / (1.0 + count))


def _ml_bond_length(symbol: str) -> float:
    """Typical M–donor distance (Å) for a given element symbol."""
    return DONOR_ML_DIST.get(symbol, DEFAULT_ML_DIST)


def _implied_bite_angle_deg(d_ab: float, ra: float, rb: float) -> float:
    """
    Estimate A–M–B bite angle assuming M sits equidistant to A and B
    with M–A ≈ ra and M–B ≈ rb using arcsin geometry. Clamped to [0, 180].
    """
    r = 0.5 * (ra + rb)
    x = np.clip(d_ab / (2.0 * r), 0.0, 1.0)
    phi = 2.0 * np.degrees(np.arcsin(x))
    return float(np.clip(phi, 0.0, 180.0))


def _angle_synergy_for_pair(d_ab: float, sym_a: str, sym_b: str,
                            ideal_deg: float, tol_deg: float) -> float:
    """
    Angle synergy in [0,1]: 1 at the ideal bite angle, fades quadratically within tolerance,
    and clamps to 0 outside.
    """
    ra = _ml_bond_length(sym_a)
    rb = _ml_bond_length(sym_b)
    phi = _implied_bite_angle_deg(d_ab, ra, rb)
    dev = abs(phi - ideal_deg)
    norm = dev / max(1e-6, tol_deg)
    return float(np.clip(1.0 - norm * norm, 0.0, 1.0))


def _pair_synergy(mol: Chem.Mol, a_idx: int, b_idx: int) -> float:
    """
    Blend steric and angle synergy into a single pairwise synergy score in [0,1].
    """
    conf = mol.GetConformer()
    pa = _np_pos(conf, a_idx)
    pb = _np_pos(conf, b_idx)
    d = float(np.linalg.norm(pa - pb))

    s_steric = _midpoint_crowding(mol, a_idx, b_idx, radius=STERIC_PARAMS["mid_radius"])
    sym_a = mol.GetAtomWithIdx(a_idx).GetSymbol()
    sym_b = mol.GetAtomWithIdx(b_idx).GetSymbol()
    s_angle = _angle_synergy_for_pair(
        d, sym_a, sym_b,
        ideal_deg=BITE_ANGLE["ideal_deg"],
        tol_deg=BITE_ANGLE["tol_deg"]
    )

    w_st = SYNERGY_WEIGHTS["steric"]
    w_an = SYNERGY_WEIGHTS["angle"]
    s = w_st * (s_steric ** 0.2) + w_an * (s_angle ** 0.2)
    # print(f'angle: {s_angle}')
    # print(f'steric: {s_steric}')

    return float(np.clip(s, 0.0, 1.0))


def _average_synergy(mol: Chem.Mol, cand_idx: int, selected: List[int]) -> float:
    """
    Average synergy of candidate versus all currently selected atoms.
    Returns 1.0 if no selected atoms are provided (should not happen in practice).
    """
    if not selected:
        return 1.0
    vals = []
    for s in selected:
        if s == cand_idx:
            continue
        vals.append(_pair_synergy(mol, s, cand_idx))
    return float(np.mean(vals)) if vals else 1.0


# -------- Patch Selection Functions --------
def _select_patch_atoms(
        mol: Chem.Mol,
        scores: List[Tuple[int, float]],
        relative_threshold: float,
        max_atoms: int,
        verbose: bool
) -> List[int]:
    """Select coordination site atoms (primary by base score; others by synergy-adjusted score)."""
    if not scores:
        raise ValueError("No heavy atoms found in molecule")

    # Sort by base score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Primary site: highest base score (no synergy applied here)
    primary_idx, primary_score = scores[0]
    patch = [primary_idx]

    if verbose:
        charges = _calculate_partial_charges(mol)
        _print_atom_info(mol, primary_idx, primary_score, charges[primary_idx], "Primary")

    # Subsequent sites: iterative single-pick per round using penalized score
    if len(scores) > 1 and max_atoms > 1:
        patch.extend(_select_subsequent_sites(
            mol=mol,
            scores=scores,
            selected=patch,  # includes primary
            primary_score=primary_score,
            threshold=relative_threshold,
            max_extra=max_atoms - 1,
            verbose=verbose
        ))

    return sorted(patch)


def _select_subsequent_sites(
        mol: Chem.Mol,
        scores: List[Tuple[int, float]],
        selected: List[int],
        primary_score: float,
        threshold: float,
        max_extra: int,
        verbose: bool
) -> List[int]:
    """
    Iteratively add at most 'max_extra' atoms.
    At each round:
      - For every non-selected candidate, compute:
            avg_synergy = average over pair synergies (candidate vs each selected atom)
            penalized_score = base_score * avg_synergy
      - Accept only the single best candidate if penalized_score >= primary_score * threshold
      - Stop when no candidate satisfies the threshold or 'max_extra' is reached
    """
    # verbose = True

    chosen: List[int] = []
    # 修复：创建 selected 的本地副本，而不是直接修改传入的列表
    current_selected = list(selected)  # 创建副本
    selected_set = set(current_selected)
    charges = _calculate_partial_charges(mol) if verbose else None

    while len(chosen) < max_extra:
        candidates = []
        for idx, base in scores:
            if idx in selected_set:
                continue
            # 使用本地副本 current_selected 而不是原始的 selected
            avg_syn = _average_synergy(mol, idx, current_selected)
            eff = base * avg_syn
            candidates.append((idx, base, avg_syn, eff))

        if not candidates:
            break

        # Pick the best by penalized score (descending)
        candidates.sort(key=lambda x: x[3], reverse=True)
        # print(candidates)
        # print(candidates[0])
        # print(threshold * primary_score)
        top_idx, top_base, top_syn, top_eff = candidates[0]

        # Threshold relative to the primary base score
        if top_eff >= threshold * primary_score:
            # 修复：只更新本地副本和 selected_set，不修改传入的 selected
            current_selected.append(top_idx)
            selected_set.add(top_idx)
            chosen.append(top_idx)

            if verbose:
                q = charges[top_idx] if charges is not None else 0.0
                _print_atom_info(mol, top_idx, top_base, q, "Secondary")
                print(f"  avg_synergy={top_syn:.3f}  penalized_score={top_eff:.3f} "
                      f"(gate={threshold * primary_score:.3f})")
        else:
            # No candidate meets the gate; stop
            break

    return chosen


def _print_atom_info(mol: Chem.Mol, idx: int, score: float, charge: float,
                     label: str, distance: Optional[float] = None):
    """Print atom information."""
    atom = mol.GetAtomWithIdx(idx)
    info = f"{label} {idx}-{atom.GetSymbol()}  base_score={score:.2f}  q={charge:+.3f}"
    if distance is not None:
        info += f"  d_to_prev={distance:.2f}Å"
    print(info)


# -------- Main Function --------
def get_patch_atoms_and_indices(
        identifier: Union[str, Atoms, Chem.Mol],
        relative_score_threshold: float = 0.9,
        max_patch_atoms: int = 2,
        total_charge: int = None,
        verbose: bool = False
) -> Tuple[Atoms, List[int]]:
    """
    Identify coordination sites.

    Parameters:
        identifier: SMILES, ASE Atoms, or RDKit Mol
        relative_score_threshold: acceptance gate for non-primary atoms based on
                                  penalized_score >= primary_base_score * threshold
        max_patch_atoms: max number of sites to return
        verbose: print details
    """
    # verbose = True
    # Handle simple ions
    if isinstance(identifier, str) and identifier.upper() in SIMPLE_IONS:
        sym = SIMPLE_IONS[identifier.upper()]
        return Atoms(sym, positions=[[0, 0, 0]]), [0]

    # 检测复杂阴离子 BF4⁻ 和 PF6⁻
    anion_result = _detect_complex_anion(identifier)
    if anion_result is not None:
        anion_name, ase_atoms, f_idx = anion_result
        if verbose:
            print(f"Detected {anion_name}⁻ anion, returning F atom at index {f_idx}")
        return ase_atoms, [f_idx]

    # Convert to RDKit Mol
    mol = _parse_input(identifier, total_charge)

    # Compute charges and base scores
    charges = _calculate_partial_charges(mol)
    scores = _compute_atom_scores(mol, charges)

    # Select coordination sites
    patch_indices = _select_patch_atoms(
        mol, scores, relative_score_threshold,
        max_patch_atoms, verbose
    )

    if verbose:
        print("Patch indices:", patch_indices)

    # Convert to ASE Atoms
    ase_atoms = mol_2_atoms(mol)

    return ase_atoms, patch_indices


# -------- Utility Functions --------
def analyze_molecule(identifier: Union[str, Atoms, Chem.Mol]) -> dict:
    """Analyze coordination properties of a molecule (base scores only)."""
    mol = _parse_input(identifier)
    charges = _calculate_partial_charges(mol)
    scores = _compute_atom_scores(mol, charges)

    result = {
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "atom_details": []
    }

    for idx, score in scores:
        atom = mol.GetAtomWithIdx(idx)
        result["atom_details"].append({
            "index": idx,
            "symbol": atom.GetSymbol(),
            "score": score,
            "charge": charges[idx],
            "hsab": HSAB_SCALE.get(atom.GetSymbol(), 0.3),
            "steric": _calculate_steric_score(atom)
        })

    result["atom_details"].sort(key=lambda x: x["score"], reverse=True)
    return result


# -------- Test Examples --------
if __name__ == "__main__":
    # Test SMILES input
    print("=== Testing SMILES input ===")
    atoms, indices = get_patch_atoms_and_indices("CCO", verbose=True)
    print(f"Result: {len(atoms)} atoms, patch indices: {indices}\n")

    # Test ASE Atoms input
    print("=== Testing ASE Atoms input ===")
    from ase.build import molecule

    water = molecule('H2O')
    atoms, indices = get_patch_atoms_and_indices(water, verbose=True)
    print(f"Result: {len(atoms)} atoms, patch indices: {indices}\n")

    # Test BF4⁻ anion
    print("=== Testing BF4⁻ anion ===")
    # Create BF4⁻ structure
    bf4_atoms = Atoms('BF4', positions=[
        [0, 0, 0],  # B
        [1.3, 0, 0],  # F1
        [-0.43, 1.23, 0],  # F2
        [-0.43, -0.61, 1.07],  # F3
        [-0.43, -0.61, -1.07]  # F4
    ])
    atoms, indices = get_patch_atoms_and_indices(bf4_atoms, verbose=True)
    print(f"Result: {len(atoms)} atoms, patch indices: {indices}\n")

    # Test PF6⁻ anion
    print("=== Testing PF6⁻ anion ===")
    # Create PF6⁻ structure
    pf6_atoms = Atoms('PF6', positions=[
        [0, 0, 0],  # P
        [1.6, 0, 0],  # F1
        [-1.6, 0, 0],  # F2
        [0, 1.6, 0],  # F3
        [0, -1.6, 0],  # F4
        [0, 0, 1.6],  # F5
        [0, 0, -1.6]  # F6
    ])
    atoms, indices = get_patch_atoms_and_indices(pf6_atoms, verbose=True)
    print(f"Result: {len(atoms)} atoms, patch indices: {indices}\n")

    # Analyze molecule
    print("=== Analyzing molecule ===")
    info = analyze_molecule("c1ccccc1O")
    print(f"Molecule has {info['num_heavy_atoms']} heavy atoms")
    print("Top 3 coordination sites:")
    for detail in info['atom_details'][:3]:
        print(f"  {detail['index']}-{detail['symbol']}: score={detail['score']:.2f}")