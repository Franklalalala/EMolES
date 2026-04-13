# This code was created and upgraded by Liu Mingkang 1660810667@qq.com

import ase
import numpy as np
from ase import Atoms, Atom
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from scipy.spatial import distance_matrix

# --- Global Parameters ---
DEFAULT_CLASH_FACTOR = 1
DEFAULT_REPULSION_POWER = 6
PUSH_AWAY_INCREMENT = 0.2
MAX_PUSH_ATTEMPTS = 3


# --- Helper Functions ---
def get_covalent_radius(symbol: str) -> float:
    try:
        return covalent_radii[atomic_numbers[symbol]]
    except KeyError:
        # print(f"Warning: Covalent radius for {symbol} not found. Using default 0.7 Å.")
        return 0.7  # Default for unknown elements


def get_bond_length(atom_1_symbol: str, atom_2_symbol: str, skin: float = 0) -> float:
    r1 = get_covalent_radius(atom_1_symbol)
    r2 = get_covalent_radius(atom_2_symbol)
    return r1 + r2 + skin


def get_nearest_neighbor(a_molecule: ase.Atoms, atom_index: int, cutoff_mult: float = None, sort_flag: bool = True) -> \
list[int]:
    if not isinstance(a_molecule, Atoms) or len(a_molecule) == 0:
        return []
    if not (0 <= atom_index < len(a_molecule)):
        raise IndexError(f"atom_index {atom_index} is out of bounds for molecule of size {len(a_molecule)}.")
    if len(a_molecule) < 2:  # No neighbors possible if only one atom
        return []

    effective_cutoff_mult = cutoff_mult if cutoff_mult is not None and cutoff_mult > 0 else 1.0

    try:
        # Ensure cutoffs are positive and non-zero for build_neighbor_list
        raw_cutoffs = natural_cutoffs(atoms=a_molecule, mult=effective_cutoff_mult)
        # ASE's natural_cutoffs might return 0 for isolated atoms or if mult is too small.
        # build_neighbor_list requires positive cutoffs.
        min_cutoff_val = 1e-3  # A very small positive floor for cutoffs
        cutoffs_for_nl = [max(c, min_cutoff_val) for c in raw_cutoffs]
        neighborList = build_neighbor_list(a_molecule, cutoffs=cutoffs_for_nl, bothways=True, self_interaction=False)
    except RuntimeError:  # Fallback if natural_cutoffs fails (e.g., with specific mult values or unusual systems)
        try:
            # Try with default ASE heuristics for cutoffs
            neighborList = build_neighbor_list(a_molecule, bothways=True, self_interaction=False)
        except Exception:  # Final fallback if neighbor list construction fails
            return []

    if neighborList is None:  # Should not happen if build_neighbor_list succeeded without error
        return []

    # Get connectivity matrix (sparse) and extract neighbors for the atom_index
    dok_matrix = neighborList.get_connectivity_matrix(sparse=True)  # DOK format is fine
    if atom_index >= dok_matrix.shape[0]:  # Should be caught by initial index check
        return []

        # Neighbors of atom_index are in the corresponding row/column
    # Using .tocoo().col gives indices of atoms connected to atom_index
    adj_array = dok_matrix[atom_index].tocoo().col

    if not adj_array.size:  # No neighbors found
        return []

    if sort_flag:
        distances_to_neighbors = [
            np.linalg.norm(a_molecule.positions[atom_index] - a_molecule.positions[neighbor_idx])
            for neighbor_idx in adj_array
        ]
        # Sort by distance, then by original index as a tie-breaker for determinism
        sorted_neighbor_indices = sorted(range(len(distances_to_neighbors)),
                                         key=lambda k: (distances_to_neighbors[k], adj_array[k]))
        return [adj_array[i] for i in sorted_neighbor_indices]

    return adj_array.tolist()  # Return as a standard list


# --- Repulsion Calculation with Pair-Specific Clash Thresholds ---
def calculate_intermolecular_repulsion(
        mol1_atoms: Atoms,
        mol2_atoms: Atoms,
        clash_factor: float = DEFAULT_CLASH_FACTOR,
        repulsion_power: int = DEFAULT_REPULSION_POWER
) -> tuple[float, bool]:
    if not isinstance(mol1_atoms, Atoms) or not isinstance(mol2_atoms, Atoms) or \
            len(mol1_atoms) == 0 or len(mol2_atoms) == 0:
        return 0.0, False  # No repulsion if one molecule is empty

    mol1_pos = mol1_atoms.get_positions()
    mol2_pos = mol2_atoms.get_positions()

    # Calculate all pair-wise distances between atoms of mol1 and mol2
    actual_distances = distance_matrix(mol1_pos, mol2_pos)

    # Determine pair-specific clash thresholds
    clash_distance_threshold_matrix = np.zeros_like(actual_distances)
    mol1_symbols = mol1_atoms.get_chemical_symbols()
    mol2_symbols = mol2_atoms.get_chemical_symbols()

    for i in range(len(mol1_atoms)):
        r1 = get_covalent_radius(mol1_symbols[i])
        for j in range(len(mol2_atoms)):
            r2 = get_covalent_radius(mol2_symbols[j])
            clash_distance_threshold_matrix[i, j] = (r1 + r2) * clash_factor

    # Check for any distance below its specific threshold
    has_clash = np.any(actual_distances < clash_distance_threshold_matrix)

    if has_clash:
        return float('inf'), True  # Return infinite repulsion for a hard clash

    # If no hard clash, calculate a finite repulsion score (e.g., Lennard-Jones repulsive term)
    epsilon = 1e-9  # Small constant to prevent division by zero if distances are extremely small
    repulsion_score = np.sum((1.0 / (actual_distances + epsilon)) ** repulsion_power)

    return repulsion_score, False


# --- Molecule Combination Utilities ---
def swap_atoms_on_copy(a_molecule: ase.Atoms, atom_index_1: int, atom_index_2: int) -> Atoms:
    if not (0 <= atom_index_1 < len(a_molecule) and 0 <= atom_index_2 < len(a_molecule)):
        raise IndexError("Atom indices are out of bounds for swapping.")

    mol_copy = a_molecule.copy()
    pos1_original = mol_copy.positions[atom_index_1].copy()
    sym1_original = mol_copy.symbols[atom_index_1]  # In ASE, symbols is a list-like object from Atoms object

    mol_copy.positions[atom_index_1] = mol_copy.positions[atom_index_2].copy()
    mol_copy.symbols[atom_index_1] = mol_copy.symbols[atom_index_2]

    mol_copy.positions[atom_index_2] = pos1_original
    mol_copy.symbols[atom_index_2] = sym1_original
    return mol_copy


def rm_and_sort_atoms(a_molecule: ase.Atoms, atom_index_to_rm: int) -> Atoms:
    if not (0 <= atom_index_to_rm < len(a_molecule)):
        raise IndexError(f"Index {atom_index_to_rm} for atom removal is out of bounds.")

    mol_copy = a_molecule.copy()

    # Get neighbors of the atom to be removed, sorted by distance.
    # This is to identify the atom that was "bonded" to the dummy atom, to place it at index 0.
    neighbors_of_atom_to_rm = get_nearest_neighbor(mol_copy, atom_index_to_rm, cutoff_mult=1.2, sort_flag=True)

    primary_bonded_atom_idx_before_del = -1
    if neighbors_of_atom_to_rm:
        primary_bonded_atom_idx_before_del = neighbors_of_atom_to_rm[0]

    # Delete the specified atom
    del mol_copy[atom_index_to_rm]

    # If a primary bonded atom was identified, adjust its index and move it to index 0
    if primary_bonded_atom_idx_before_del != -1:
        # Adjust index due to deletion: if it was after the deleted atom, its index decreases by 1.
        primary_bonded_atom_idx_after_del = primary_bonded_atom_idx_before_del
        if primary_bonded_atom_idx_before_del > atom_index_to_rm:
            primary_bonded_atom_idx_after_del -= 1

        # Ensure the adjusted index is still valid and not already 0
        if 0 <= primary_bonded_atom_idx_after_del < len(mol_copy) and primary_bonded_atom_idx_after_del != 0:
            mol_copy = swap_atoms_on_copy(mol_copy, 0, primary_bonded_atom_idx_after_del)

    return mol_copy


def combine_2_mols(molecule_1: Atoms, molecule_2: Atoms,
                   tgt_atom_1_index: int,  # Attachment atom index on molecule_1
                   tgt_atom_2_indices: list[int],  # Patch atom indices on molecule_2
                   sample_times: int = 1500, skin: float = 0,
                   cutoff_mult: float = 1.2,  # For finding internal neighbors for rotation
                   rotation_times: int = 3,  # Max number of neighbor axes to use for rotation
                   molecule_1_is_primary: bool = False  # Hint if mol1 must be the fixed "main" molecule
                   ) -> Atoms:
    mol1_c, mol2_c = molecule_1.copy(), molecule_2.copy()

    if not mol1_c and not mol2_c: return Atoms()
    if not mol1_c: return mol2_c
    if not mol2_c: return mol1_c

    # Determine main molecule (fixed) and sub_molecule (to be placed and rotated)
    if molecule_1_is_primary or len(mol1_c) >= len(mol2_c):
        main_mol, sub_mol = mol1_c, mol2_c
        main_attach_idx = tgt_atom_1_index
        sub_patch_indices = tgt_atom_2_indices  # These are indices on the original sub_mol (mol2_c)
    else:  # molecule_2 is larger and molecule_1 is not forced as primary
        main_mol, sub_mol = mol2_c, mol1_c
        # tgt_atom_2_indices should contain the attachment index for mol2 if it's main
        # Assuming tgt_atom_2_indices[0] is the primary attachment point if mol2 becomes main
        main_attach_idx = tgt_atom_2_indices[0] if tgt_atom_2_indices else 0
        sub_patch_indices = [tgt_atom_1_index]  # Index on original sub_mol (mol1_c)

    if not (0 <= main_attach_idx < len(main_mol)):
        raise IndexError(f"Main attachment index {main_attach_idx} out of bounds for main molecule.")
    if not sub_patch_indices or not all(0 <= i < len(sub_mol) for i in sub_patch_indices):
        raise IndexError(f"Sub-molecule patch indices {sub_patch_indices} are invalid.")

    main_mol_target_atom = main_mol[main_attach_idx]
    sub_mol_primary_attach_idx_on_sub = sub_patch_indices[0]  # Primary atom in patch on sub_mol
    sub_mol_primary_attach_sym = sub_mol.symbols[sub_mol_primary_attach_idx_on_sub]

    # Anchor for placement on sub_mol is the centroid of its patch atoms (in sub_mol's local coords)
    sub_mol_patch_coords_local = sub_mol.get_positions()[sub_patch_indices]
    sub_mol_patch_centroid_local = np.mean(sub_mol_patch_coords_local, axis=0)

    ideal_placement_dist = get_bond_length(main_mol_target_atom.symbol, sub_mol_primary_attach_sym, skin)

    best_translation_vector = None
    min_repulsion_at_best_translation = float('inf')
    found_non_clashing_translation = False

    # --- Translational Sampling ---
    for _ in range(sample_times):
        random_direction = np.random.randn(3)
        if np.linalg.norm(random_direction) < 1e-6: random_direction = np.array([1., 0., 0.])  # Avoid zero vector
        random_direction /= np.linalg.norm(random_direction)

        current_push_offset = 0.0
        found_non_clashing_this_direction = False

        for push_attempt in range(MAX_PUSH_ATTEMPTS + 1):  # +1 to try ideal_dist first
            current_target_dist = ideal_placement_dist + current_push_offset

            # Desired world position for sub_mol's patch centroid
            candidate_sub_mol_centroid_world_pos = main_mol_target_atom.position + current_target_dist * random_direction
            # Translation vector needed to move sub_mol's patch centroid to this candidate position
            trial_translation_vec = candidate_sub_mol_centroid_world_pos - sub_mol_patch_centroid_local

            temp_sub_mol_translated = sub_mol.copy()
            temp_sub_mol_translated.translate(trial_translation_vec)

            repulsion, clash_detected = calculate_intermolecular_repulsion(main_mol, temp_sub_mol_translated)

            if not clash_detected:
                if repulsion < min_repulsion_at_best_translation:
                    min_repulsion_at_best_translation = repulsion
                    best_translation_vector = trial_translation_vec
                found_non_clashing_translation = True
                found_non_clashing_this_direction = True
                break  # Found non-clashing for this direction, stop pushing

            if push_attempt < MAX_PUSH_ATTEMPTS:  # If still clashing and more push attempts left
                current_push_offset += PUSH_AWAY_INCREMENT
            elif repulsion < min_repulsion_at_best_translation:  # Last push attempt, still clashing, but maybe best clashing score
                min_repulsion_at_best_translation = repulsion  # This will be 'inf'
                best_translation_vector = trial_translation_vec

        if found_non_clashing_this_direction and min_repulsion_at_best_translation == 0:  # Ideal (no clash, zero repulsion)
            break  # Exit translational sampling early

    if best_translation_vector is None:  # Fallback if all samples clashed (should be rare if MAX_PUSH_ATTEMPTS is effective)
        fallback_direction = np.array([0, 0, 1.0])  # Arbitrary direction
        best_translation_vector = (
                                              main_mol_target_atom.position + ideal_placement_dist * fallback_direction) - sub_mol_patch_centroid_local

    sub_mol.translate(best_translation_vector)  # Apply the best translation found

    # World position of sub_mol's primary attachment atom (this will be the center for rotation)
    sub_mol_primary_attach_atom_world_pos = sub_mol.positions[sub_mol_primary_attach_idx_on_sub]

    if len(sub_mol) == 1:  # No rotation needed for single-atom sub-molecule
        main_mol.extend(sub_mol)
        return main_mol

    # --- Rotational Sampling ---
    rotation_center = sub_mol_primary_attach_atom_world_pos
    current_best_rotation_score, _ = calculate_intermolecular_repulsion(main_mol, sub_mol)  # Score after translation
    final_sub_mol_orientation = sub_mol.copy()  # Start with translated orientation

    # Get internal neighbors of the primary attachment atom on the sub_molecule (for defining rotation axes)
    internal_neighbors_of_attach_atom = get_nearest_neighbor(sub_mol, sub_mol_primary_attach_idx_on_sub, cutoff_mult,
                                                             True)
    num_neighbor_axes_to_try = min(len(internal_neighbors_of_attach_atom),
                                   rotation_times) if internal_neighbors_of_attach_atom else 0

    for _rot_trial in range(sample_times):  # Outer loop for randomizing orientation broadly
        trial_orientation_sub_mol = sub_mol.copy()  # Start from the translated state for each broad trial

        # Apply an initial random spin, or spin around specific axes
        if num_neighbor_axes_to_try == 0:  # e.g. diatomic, or no neighbors found for rotation_times limit
            rand_axis = np.random.randn(3)
            if np.linalg.norm(rand_axis) < 1e-6: rand_axis = np.array([1., 0., 0.])
            rand_axis /= np.linalg.norm(rand_axis)
            trial_orientation_sub_mol.rotate(np.random.uniform(0, 360), rand_axis, center=rotation_center)
        else:  # Spin around axes defined by internal neighbors
            for i_ax in range(num_neighbor_axes_to_try):
                neighbor_idx_on_sub = internal_neighbors_of_attach_atom[i_ax]

                # Vector from neighbor to primary attachment atom (on sub_mol)
                # This defines a bond-like axis for rotation
                p_atom_pos_local = trial_orientation_sub_mol.positions[sub_mol_primary_attach_idx_on_sub]
                n_atom_pos_local = trial_orientation_sub_mol.positions[neighbor_idx_on_sub]
                rotation_axis_vec = p_atom_pos_local - n_atom_pos_local

                if np.linalg.norm(rotation_axis_vec) > 1e-6:
                    trial_orientation_sub_mol.rotate(np.random.uniform(0, 360), rotation_axis_vec,
                                                     center=rotation_center)

        # Evaluate this new orientation
        repulsion_trial_orientation, clash_trial_orientation = calculate_intermolecular_repulsion(main_mol,
                                                                                                  trial_orientation_sub_mol)

        if not clash_trial_orientation and repulsion_trial_orientation < current_best_rotation_score:
            current_best_rotation_score = repulsion_trial_orientation
            final_sub_mol_orientation = trial_orientation_sub_mol.copy()
            if current_best_rotation_score == 0:  # Perfect orientation found
                break
        elif clash_trial_orientation and current_best_rotation_score == float(
                'inf'):  # If current best is clashing, any clashing is fine to compare
            if repulsion_trial_orientation < current_best_rotation_score:  # (inf < inf is false, but useful if penalties were used)
                current_best_rotation_score = repulsion_trial_orientation  # Still inf
                final_sub_mol_orientation = trial_orientation_sub_mol.copy()

        if current_best_rotation_score == 0 and _rot_trial > 10:  # Early exit if perfect found quickly
            break

    main_mol.extend(final_sub_mol_orientation)
    return main_mol


def combine_2_mols_with_dummy(mol1: Atoms, mol2: Atoms,
                              dummy1_idx: int, dummy2_idx: int,
                              **kwargs) -> Atoms:
    """
    Combines two molecules after removing a dummy atom from each.
    The atom previously bonded to the dummy atom in each molecule becomes the new attachment point (moved to index 0).
    """
    m1_processed = rm_and_sort_atoms(mol1, dummy1_idx)  # Atom bonded to dummy1_idx is now at index 0
    m2_processed = rm_and_sort_atoms(mol2, dummy2_idx)  # Atom bonded to dummy2_idx is now at index 0

    if not m1_processed and not m2_processed: return Atoms()
    if not m1_processed: return m2_processed
    if not m2_processed: return m1_processed

    # Attachment points are now at index 0 for both processed molecules
    # molecule_1_is_primary=False lets combine_2_mols decide main/sub based on size.
    # If a specific main/sub behavior is needed, this could be a kwarg.
    return combine_2_mols(m1_processed, m2_processed, 0, [0], molecule_1_is_primary=False, **kwargs)


if __name__ == '__main__':
    from ase.visualize import view
    from ase.build import molecule as ase_molecule_builder

    print(f"--- Testing CombineMols3D_v2.py ---")
    print(f"Global Defaults: CLASH_FACTOR={DEFAULT_CLASH_FACTOR}, REPULSION_POWER={DEFAULT_REPULSION_POWER}")

    # Test 1: calculate_intermolecular_repulsion
    print("\n--- Test: calculate_intermolecular_repulsion ---")
    li_atom = Atoms('Li', positions=[[0, 0, 0]])
    f_atom = Atoms('F', positions=[[1.0, 0, 0]])  # Distance = 1.0 Å

    r_Li = get_covalent_radius('Li')
    r_F = get_covalent_radius('F')
    expected_clash_dist_threshold = (r_Li + r_F) * DEFAULT_CLASH_FACTOR
    print(f"Li covalent radius: {r_Li:.3f} Å, F covalent radius: {r_F:.3f} Å")
    print(
        f"Li-F expected clash distance threshold (@factor {DEFAULT_CLASH_FACTOR}): {expected_clash_dist_threshold:.3f} Å")

    rep, clash = calculate_intermolecular_repulsion(li_atom, f_atom)
    print(
        f"  Actual Li-F distance 1.0 Å: Repulsion Score={rep:.2e}, Clash Detected={clash} (Expected Clash if 1.0 < threshold: {1.0 < expected_clash_dist_threshold})")

    f_atom_no_clash_pos = expected_clash_dist_threshold + 0.01
    f_atom.positions[0, 0] = f_atom_no_clash_pos
    rep, clash = calculate_intermolecular_repulsion(li_atom, f_atom)
    print(
        f"  Actual Li-F distance {f_atom_no_clash_pos:.3f} Å: Repulsion Score={rep:.2e}, Clash Detected={clash} (Expected Clash: False)")

    # Test 2: combine_2_mols - e.g., water and methane
    print("\n--- Test: combine_2_mols (H2O + CH4) ---")
    water = ase_molecule_builder('H2O')  # O index 0, H indices 1, 2
    methane = ase_molecule_builder('CH4')  # C index 0, H indices 1,2,3,4

    # Attach one of water's H (index 1) to methane's C (index 0)
    # Let combine_2_mols decide main/sub (methane will be main due to size)
    # So, main_attach_idx will be methane's C (0), sub_patch_indices will be water's H ([1])
    try:
        combined_mol_h2o_ch4 = combine_2_mols(water, methane,
                                              tgt_atom_1_index=1,  # H on water
                                              tgt_atom_2_indices=[0],  # C on methane
                                              skin=0.1, sample_times=100, rotation_times=2)  # fewer samples for test
        print(f"  H2O + CH4 combined. Total atoms: {len(combined_mol_h2o_ch4)}")
        # view(combined_mol_h2o_ch4) # Uncomment to view
    except Exception as e:
        print(f"  Error combining H2O and CH4: {e}")

    # Test 3: combine_2_mols_with_dummy
    print("\n--- Test: combine_2_mols_with_dummy (Ethane from two CH3X) ---")
    # Create two methyl groups with a dummy atom X
    ch3x1 = Atoms('CHHHX', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])  # X at index 4
    ch3x2 = Atoms('CHHHX', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])  # X at index 4
    # X is bonded to C (index 0). rm_and_sort_atoms will make C index 0.
    try:
        ethane = combine_2_mols_with_dummy(ch3x1, ch3x2,
                                           dummy1_idx=4, dummy2_idx=4,
                                           skin=0.0, sample_times=100)
        print(f"  Ethane from CH3X+CH3X. Total atoms: {len(ethane)} (Expected 8 for C2H6)")
        # view(ethane) # Uncomment to view
    except Exception as e:
        print(f"  Error combining CH3X dummies: {e}")