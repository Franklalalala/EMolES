"""
emoles/build/patch_picker.py

Updated with robust handling for:
1. Single-atom ASE objects (skips RDKit overhead).
2. Anion topology inference failures (FSI/TFSI charge fallbacks).
"""

from typing import Tuple, List, Union, Optional

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import BlockLogs

# =============================================================================
# 1. CONSTANTS & HYPERPARAMETERS
# =============================================================================

HSAB_SCALE = {
    "O": 1.00, "F": 0.85, "N": 0.80, "Cl": 0.70,
    "S": 0.60, "Br": 0.50, "I": 0.40, "P": 0.40,
}

WEIGHTS = dict(
    charge=0.5,
    hsab=1.5,
    steric=2.0,
)

STERIC_PARAMS = dict(mid_radius=2.0)
SYNERGY_WEIGHTS = dict(steric=0.7, angle=0.3)
BITE_ANGLE = dict(ideal_deg=90.0, tol_deg=20.0)

SIMPLE_IONS = {
    "LI": "Li", "NA": "Na", "K": "K",
    "MG": "Mg", "CA": "Ca", "ZN": "Zn",
}



DONOR_ML_DIST = {"O": 2.10, "F": 2.00, "N": 2.15, "S": 2.30, "Cl": 2.40}
DEFAULT_ML_DIST = 2.20

# =============================================================================
# [UPDATE 1] Replace the old COMPLEX_ANIONS dictionary with this block
# =============================================================================

COMPLEX_ANIONS = {
    "BF4": {"center": "B", "ligand": "F", "count": 4, "return_count": 1},  # Tetrahedral, monodentate is fine
    "PF6": {"center": "P", "ligand": "F", "count": 6, "return_count": 1},  # Octahedral, monodentate is fine
    "ClO4": {"center": "Cl", "ligand": "O", "count": 4, "return_count": 1},  # Tetrahedral
    "NO3": {"center": "N", "ligand": "O", "count": 3, "return_count": 2},  # [CRITICAL FIX] Planar, Bidentate required
}


# =============================================================================
# [UPDATE 2] Replace the _detect_complex_anion function
# =============================================================================

def _detect_complex_anion(identifier: Union[str, Atoms, Chem.Mol]) -> Optional[Tuple[str, Atoms, List[int]]]:
    """
    Detects BF4-, PF6-, ClO4-, NO3- and returns representative binding atoms.

    Updated Logic:
    - Now returns a LIST of indices based on 'return_count'.
    - Specifically allows NO3 to return 2 Oxygen indices for bidentate coordination.
    """
    if isinstance(identifier, str):
        try:
            mol = Chem.MolFromSmiles(identifier)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            ase_atoms = mol_2_atoms(mol)
        except Exception:
            return None
    elif isinstance(identifier, Atoms):
        ase_atoms = identifier
    elif isinstance(identifier, Chem.Mol):
        ase_atoms = mol_2_atoms(identifier)
    else:
        return None

    symbols = ase_atoms.get_chemical_symbols()

    for anion_name, pattern in COMPLEX_ANIONS.items():
        center_symbol = pattern["center"]
        ligand_symbol = pattern["ligand"]
        expected_count = pattern["count"]
        return_count = pattern.get("return_count", 1)

        # Check composition (e.g. 1 N and 3 O for NO3)
        if symbols.count(center_symbol) == 1 and symbols.count(ligand_symbol) == expected_count:
            total_non_h = sum(1 for s in symbols if s != "H")

            # Ensure no other heavy atoms exist
            if total_non_h == (1 + expected_count):
                # Collect ALL matching ligand indices
                ligand_indices = []
                for idx, symbol in enumerate(symbols):
                    if symbol == ligand_symbol:
                        ligand_indices.append(idx)

                # Return the requested number of anchor points
                # For NO3, this returns 2 Oxygens; for PF6, returns 1 Fluorine.
                return anion_name, ase_atoms, ligand_indices[:return_count]

    return None


# =============================================================================
# [UPDATE 3] Replace the get_patch_atoms_and_indices function
# =============================================================================


# =============================================================================
# 2. CHEMISTRY CONVERSION FUNCTIONS
# =============================================================================

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
    Convert ASE Atoms to RDKit Mol via in-memory XYZ block.

    [INTEGRATED FIX]:
    - Tries to determine bonds using the requested charge.
    - If that fails (common with complex anions like FSI-), falls back to
      neutral charge (0), and finally to None (letting RDKit decide).
    """
    xyz_block = atoms_to_plain_xyz_block(atoms)

    # Create raw Mol from XYZ (no bonds yet)
    with BlockLogs():
        mol = Chem.MolFromXYZBlock(xyz_block)

    if mol is None:
        raise ValueError("RDKit failed to generate molecule from XYZ data")

    mol = Chem.Mol(mol)  # make a writable copy

    # If it's a single atom, bonding is irrelevant, but we return the mol.
    # If >1 atoms, we need topology.
    if mol.GetNumAtoms() > 1:
        success = False

        # Attempt 1: Requested Charge
        try:
            with BlockLogs():
                DetermineBonds(mol, charge=charge, useHueckel=(charge == 0))
            success = True
        except ValueError:
            pass

        # Attempt 2: Fallback to Neutral (often more stable for topology guessing)
        if not success:
            try:
                # print(f"DEBUG: RDKit charge {charge} failed, retrying with charge=0")
                with BlockLogs():
                    DetermineBonds(mol, charge=0, useHueckel=False)
                success = True
            except ValueError:
                pass

        # Attempt 3: Fallback to None (RDKit decides best fit)
        if not success:
            try:
                # print(f"DEBUG: RDKit charge 0 failed, retrying with charge=None")
                with BlockLogs():
                    DetermineBonds(mol, charge=None, useHueckel=True)
                success = True
            except Exception:
                pass  # If this fails, the molecule might be broken, but we return what we have.

    # Sanitize and Update Cache
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True,
        )
    except Exception:
        pass

    return mol


def mol_2_atoms(mol: Chem.Mol) -> Atoms:
    """Convert RDKit Mol to ASE Atoms."""
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]
    return Atoms(symbols=symbols, positions=positions)


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================



def _parse_input(identifier: Union[str, Atoms, Chem.Mol], total_charge=None) -> Chem.Mol:
    """Unified input handling with robust property caching."""
    if isinstance(identifier, str):
        # SMILES Input
        mol = Chem.MolFromSmiles(identifier)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {identifier}")
        mol = Chem.AddHs(mol)

        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))

        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

        mol.UpdatePropertyCache(strict=False)
        return mol

    elif isinstance(identifier, Atoms):
        # ASE Input
        if total_charge is None:
            # Try to infer charge from ASE object
            if identifier.has("initial_charges"):
                charge_sum = identifier.get_initial_charges().sum()
                total_charge = int(round(charge_sum))
            elif identifier.has("charge"):
                try:
                    total_charge = int(identifier.charge)
                except Exception:
                    total_charge = 0
            else:
                try:
                    total_charge = int(identifier.info.get("charge", 0))
                except Exception:
                    total_charge = 0

        # Pass inferred charge to atom_2_mol (which now handles fallbacks)
        return atom_2_mol(identifier, charge=total_charge if total_charge is not None else 0)

    elif isinstance(identifier, Chem.Mol):
        # RDKit Mol Input
        if identifier.GetNumConformers() == 0:
            mol = Chem.AddHs(identifier)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol.UpdatePropertyCache(strict=False)
            return mol

        identifier.UpdatePropertyCache(strict=False)
        return identifier

    else:
        raise TypeError(f"Unsupported input type: {type(identifier)}")


# =============================================================================
# 4. SCORING & SYNERGY FUNCTIONS
# =============================================================================

def _calculate_steric_score(atom: Chem.Atom) -> float:
    heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
    return 1.0 / (1 + heavy_neighbors)


def _compute_atom_scores(mol: Chem.Mol) -> List[Tuple[int, float]]:
    """Score atoms using tuned weights."""
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # Partial Charges
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            charges = [mp.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
        else:
            raise ValueError("MMFF missing")
    except Exception:
        # Gasteiger Fallback
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            try:
                charge = atom.GetDoubleProp("_GasteigerCharge")
                if np.isnan(charge):
                    charge = 0.0
            except Exception:
                charge = 0.0
            charges.append(charge)

    scores: List[Tuple[int, float]] = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:  # skip H
            continue

        q = -charges[i]
        hsab = HSAB_SCALE.get(atom.GetSymbol(), 0.3)
        steric = _calculate_steric_score(atom)

        total = (
                WEIGHTS["charge"] * q
                + WEIGHTS["hsab"] * hsab
                + WEIGHTS["steric"] * steric
        )
        scores.append((i, total))
    return scores


def _np_pos(conf: Chem.Conformer, idx: int) -> np.ndarray:
    p = conf.GetAtomPosition(idx)
    return np.array([p.x, p.y, p.z], dtype=float)


def _pair_synergy(mol: Chem.Mol, a_idx: int, b_idx: int) -> float:
    conf = mol.GetConformer()
    pa = _np_pos(conf, a_idx)
    pb = _np_pos(conf, b_idx)
    d = float(np.linalg.norm(pa - pb))

    # Steric check at midpoint
    mid = 0.5 * (pa + pb)
    count = 0
    for j in range(mol.GetNumAtoms()):
        if j == a_idx or j == b_idx:
            continue
        if mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        pj = _np_pos(conf, j)
        if np.linalg.norm(pj - mid) < STERIC_PARAMS["mid_radius"]:
            count += 1
    s_steric = 1.0 / (1.0 + count)

    # Bite angle proxy
    sym_a = mol.GetAtomWithIdx(a_idx).GetSymbol()
    sym_b = mol.GetAtomWithIdx(b_idx).GetSymbol()
    ra = DONOR_ML_DIST.get(sym_a, DEFAULT_ML_DIST)
    rb = DONOR_ML_DIST.get(sym_b, DEFAULT_ML_DIST)

    x_clamped = np.clip(d / (ra + rb), 0.0, 1.0)
    phi = 2.0 * np.degrees(np.arcsin(x_clamped))

    dev = abs(phi - BITE_ANGLE["ideal_deg"])
    norm = dev / max(1e-6, BITE_ANGLE["tol_deg"])
    s_angle = float(np.clip(1.0 - norm * norm, 0.0, 1.0))

    return float(
        np.clip(
            SYNERGY_WEIGHTS["steric"] * (s_steric ** 0.2)
            + SYNERGY_WEIGHTS["angle"] * (s_angle ** 0.2),
            0,
            1,
        )
    )


def _average_synergy(mol: Chem.Mol, cand_idx: int, selected: List[int]) -> float:
    if not selected:
        return 1.0
    vals = [_pair_synergy(mol, s, cand_idx) for s in selected]
    return float(np.mean(vals))


# =============================================================================
# 5. MAIN EXPORT FUNCTION
# =============================================================================
def get_patch_atoms_and_indices(
        identifier: Union[str, Atoms, Chem.Mol],
        relative_score_threshold: float = 0.8,
        max_patch_atoms: int = 2,
        total_charge: int = None,
        verbose: bool = False,
) -> Tuple[Atoms, List[int]]:
    """
    Given a molecule (SMILES / ASE Atoms / RDKit Mol), returns:
    - ASE Atoms version of the molecule.
    - List of atom indices best suited to coordinate with a metal center.

    [INTEGRATED FIX]:
    - Short-circuits for single-atom inputs (e.g., Li+) to prevent RDKit errors.
    - Properly handles multi-atom anchors for complex anions (e.g. NO3 bidentate).
    """

    # [HOTFIX INTEGRATION]: Single Atom Bypass
    if isinstance(identifier, Atoms) and len(identifier) == 1:
        if verbose:
            print("Detected single atom input, returning immediately.")
        return identifier, [0]

    # 1. Handle Simple Ions (Strings like "Li")
    if isinstance(identifier, str) and identifier.upper() in SIMPLE_IONS:
        sym = SIMPLE_IONS[identifier.upper()]
        return Atoms(sym, positions=[[0.0, 0.0, 0.0]]), [0]

    # 2. Handle known Complex Anions (e.g., BF4-, NO3-)
    anion_result = _detect_complex_anion(identifier)
    if anion_result is not None:
        anion_name, ase_atoms, coord_indices = anion_result
        if verbose:
            print(f"Detected {anion_name}⁻ anion, forcing indices {coord_indices}")
        # Directly return the hardcoded anchor indices (e.g. [0, 1] for NO3)
        return ase_atoms, coord_indices

    # 3. Parse Input -> RDKit Mol
    mol = _parse_input(identifier, total_charge)

    # 4. Compute Scores (Generic Logic)
    scores = _compute_atom_scores(mol)
    scores.sort(key=lambda x: x[1], reverse=True)

    if not scores:
        return mol_2_atoms(mol), []

    # 5. Selection Loop (Patch Building)
    primary_idx, primary_score = scores[0]
    patch = [primary_idx]
    current_selected = list(patch)
    selected_set = set(patch)

    if verbose:
        print(f"Primary site: {primary_idx} (Score: {primary_score:.3f})")

    while len(patch) < max_patch_atoms and len(scores) > len(patch):
        candidates = []
        for idx, base in scores:
            if idx in selected_set:
                continue

            avg_syn = _average_synergy(mol, idx, current_selected)
            eff = base * avg_syn
            candidates.append((idx, base, avg_syn, eff))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[3], reverse=True)
        top_idx, top_base, top_syn, top_eff = candidates[0]

        gate_value = primary_score * relative_score_threshold
        if top_eff >= gate_value:
            patch.append(top_idx)
            current_selected.append(top_idx)
            selected_set.add(top_idx)
            if verbose:
                print(f"Added secondary: {top_idx}")
        else:
            break

    patch.sort()
    return mol_2_atoms(mol), patch
