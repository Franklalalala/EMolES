import os
import shutil
from pathlib import Path
from emoles.constant import convention_dict
import numpy as np


def setup_output_directory(output_path):
    """
    Set up the output directory.

    Args:
        output_path: Path for output files

    """
    output_dir = Path(output_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    return

def setup_db_path(db_path):
    """
    Set up the database file.

    Args:
        db_path: Path to the ASE database file

    Returns:
        absolute_db_path: Absolute path to the database file
    """
    absolute_db_path = os.path.abspath(db_path)

    # Clean up existing directories/files
    if os.path.exists(absolute_db_path):
        os.remove(absolute_db_path)

    return absolute_db_path


def cut_and_cal_matrix(full_matrix, atom_in_mo_indices):
    atom_indeces = sorted(set(atom_in_mo_indices))
    atom_positions = {atom: [i for i, x in enumerate(atom_in_mo_indices) if x == atom] for atom in atom_indeces}
    diag_mae_list = []
    non_diag_mae_list = []

    # Extract blocks for each pair of atoms
    for ii, i in enumerate(atom_indeces):
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


def format_number(num):
    """Format number based on magnitude"""
    abs_num = abs(num)
    if abs_num >= 1:
        return f"{num:.2f}"
    else:
        # Convert to string with 3 significant figures
        return f"{num:.3g}"


def vec_cosine_similarity(a, b):
    # Calculate the dot product and norms
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.abs(dot_product / (norm_a * norm_b))


def get_shifted_ham(predicted_ham, label_ham, overlap):
    diff_ham = predicted_ham - label_ham
    diagonal_diff = np.diagonal(diff_ham)
    mean_diagonal = np.mean(diagonal_diff)
    shifted_label_ham = label_ham + mean_diagonal * overlap
    return shifted_label_ham


def get_mo_occ(full_len: int, occ_len: int):
    mo_occ = np.zeros(full_len)
    mo_occ[:occ_len] = 2
    return mo_occ


def matrix_transform(matrices, atoms, convention='pyscf_631G'):
    conv = convention_dict[convention]
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
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


def get_atom_in_mo_indices(atomic_numbers, convention_name, convention_dict):
    """
    输入：
      - atomic_numbers: list of ints, e.g. [6, 1, 8]
      - convention_name: str, e.g. 'pyscf_def2svp'
      - convention_dict: dict-like mapping convention_name -> Namespace/dict, like your example

    输出：
      - atom_in_mo_indices: list of ints, length == total number of basis functions;
        每个元素是该 MO 对应的原子索引（按传入 atomic_numbers 的顺序，从 0 开始）
    """
    # 获取 convention entry（支持 Namespace 或 dict）
    conv = convention_dict.get(convention_name) if isinstance(convention_dict, dict) else convention_dict[convention_name]
    # try both attribute and dict access
    atom_to_orbitals_map = getattr(conv, 'atom_to_orbitals_map', None)
    if atom_to_orbitals_map is None:
        atom_to_orbitals_map = conv.get('atom_to_orbitals_map') if isinstance(conv, dict) else None
    if atom_to_orbitals_map is None:
        raise KeyError(f"convention {convention_name} 中找不到 'atom_to_orbitals_map'")

    # 如何把壳字母展开为轨道数（常见：s=1, p=3, d=5, f=7）
    shell_size = {'s': 1, 'p': 3, 'd': 5, 'f': 7}

    atom_in_mo_indices = []
    for atom_idx, Z in enumerate(atomic_numbers):
        if Z not in atom_to_orbitals_map:
            raise KeyError(f"convention {convention_name} 中没有原子 {Z} 的映射（atom_to_orbitals_map）")
        shells = atom_to_orbitals_map[Z]  # e.g. 'ssp' or 'sssppd'
        # 允许 shells 是字符串或 list（若 list 则元素像 's','p'）
        if isinstance(shells, (list, tuple)):
            shell_chars = ''.join(shells)
        else:
            shell_chars = str(shells)

        local_orbital_count = 0
        for ch in shell_chars:
            ch_low = ch.lower()
            if ch_low not in shell_size:
                raise ValueError(f"未知的轨道字符 '{ch}'（仅支持 s,p,d,f）")
            local_orbital_count += shell_size[ch_low]

        atom_in_mo_indices.extend([atom_idx] * local_orbital_count)

    return atom_in_mo_indices


def cut_matrix(full_matrix, atom_in_mo_indices, threshold=1e-8):
    partitioned_blocks = {}
    atom_indeces = sorted(set(atom_in_mo_indices))
    atom_positions = {atom: [i for i, x in enumerate(atom_in_mo_indices) if x == atom] for atom in atom_indeces}

    # Extract blocks for each pair of atoms
    for ii, i in enumerate(atom_indeces):
        for j in atom_indeces[ii:]:
            key = f"{i}_{j}_0_0_0"
            rows = atom_positions[i]
            cols = atom_positions[j]
            block = full_matrix[np.ix_(rows, cols)]
            if np.max(np.abs(block)) > threshold:
                partitioned_blocks[key] = block
    return partitioned_blocks
