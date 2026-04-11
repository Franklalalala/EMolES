import numpy as np

from emoles.constant import convention_dict


def cut_and_cal_matrix(full_matrix, atom_in_mo_indices):
    atom_indeces = sorted(set(atom_in_mo_indices))
    atom_positions = {
        atom: [i for i, x in enumerate(atom_in_mo_indices) if x == atom]
        for atom in atom_indeces
    }
    diag_mae_list = []
    non_diag_mae_list = []

    for i in atom_indeces:
        for j in atom_indeces:
            rows = atom_positions[i]
            cols = atom_positions[j]
            block_mae = float(np.mean(full_matrix[np.ix_(rows, cols)]))
            if i == j:
                diag_mae_list.append(block_mae)
            else:
                non_diag_mae_list.append(block_mae)

    diag_mae = np.mean(np.array(diag_mae_list))
    non_diag_mae = np.mean(np.array(non_diag_mae_list))
    return diag_mae, non_diag_mae


def get_shifted_ham(predicted_ham, label_ham, overlap):
    diff_ham = predicted_ham - label_ham
    diagonal_diff = np.diagonal(diff_ham)
    mean_diagonal = np.mean(diagonal_diff)
    shifted_label_ham = label_ham + mean_diagonal * overlap
    return shifted_label_ham


def matrix_transform(matrices, atoms, convention="pyscf_631G"):
    conv = convention_dict[convention]
    orbitals = ""
    orbitals_order = []
    for atom in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[atom]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[atom]]

    transform_indices = []
    transform_signs = []
    for orbital in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orbital]
        map_sign = conv.orbital_sign_map[orbital]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    matrices_new = matrices[..., transform_indices, :]
    matrices_new = matrices_new[..., :, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


def generate_molecule_transform_indices(atom_types, atom_to_transform_indices):
    molecule_transform_indices = []
    atom_in_mo_indices = []
    current_offset = 0

    for atomic_idx, atom_type in enumerate(atom_types):
        atom_indices = atom_to_transform_indices[atom_type]
        adjusted_indices = [index + current_offset for index in atom_indices]
        molecule_transform_indices.extend(adjusted_indices)
        atom_in_mo_indices.extend([atomic_idx] * len(atom_indices))
        current_offset += max(atom_indices) + 1

    return molecule_transform_indices, atom_in_mo_indices


def get_atom_in_mo_indices(atomic_numbers, convention_name, convention_map):
    """
    Map each AO/MO basis-function slot back to its atom index.
    """
    conv = (
        convention_map.get(convention_name)
        if isinstance(convention_map, dict)
        else convention_map[convention_name]
    )
    atom_to_orbitals_map = getattr(conv, "atom_to_orbitals_map", None)
    if atom_to_orbitals_map is None:
        atom_to_orbitals_map = (
            conv.get("atom_to_orbitals_map") if isinstance(conv, dict) else None
        )
    if atom_to_orbitals_map is None:
        raise KeyError(
            f"convention {convention_name} missing 'atom_to_orbitals_map'"
        )

    shell_size = {"s": 1, "p": 3, "d": 5, "f": 7}
    atom_in_mo_indices = []
    for atom_idx, atomic_number in enumerate(atomic_numbers):
        if atomic_number not in atom_to_orbitals_map:
            raise KeyError(
                f"convention {convention_name} missing mapping for atom {atomic_number}"
            )
        shells = atom_to_orbitals_map[atomic_number]
        shell_chars = "".join(shells) if isinstance(shells, (list, tuple)) else str(shells)

        local_orbital_count = 0
        for shell in shell_chars:
            shell_key = shell.lower()
            if shell_key not in shell_size:
                raise ValueError(
                    f"Unknown orbital character '{shell}', only s/p/d/f are supported"
                )
            local_orbital_count += shell_size[shell_key]

        atom_in_mo_indices.extend([atom_idx] * local_orbital_count)

    return atom_in_mo_indices


def cut_matrix(full_matrix, atom_in_mo_indices, threshold=1e-8):
    partitioned_blocks = {}
    atom_indeces = sorted(set(atom_in_mo_indices))
    atom_positions = {
        atom: [i for i, x in enumerate(atom_in_mo_indices) if x == atom]
        for atom in atom_indeces
    }

    for ii, i in enumerate(atom_indeces):
        for j in atom_indeces[ii:]:
            key = f"{i}_{j}_0_0_0"
            rows = atom_positions[i]
            cols = atom_positions[j]
            block = full_matrix[np.ix_(rows, cols)]
            if np.max(np.abs(block)) > threshold:
                partitioned_blocks[key] = block
    return partitioned_blocks
