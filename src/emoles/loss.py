import os
import time
import pickle
import json

import numpy as np
import pyscf
import torch
from ase.io import write
from ase.db.core import connect
from ase.units import Hartree
from pyscf.scf.hf import make_rdm1
from pyscf import gto, dft, tools
from tqdm import tqdm
from emoles.pyscf import generate_cube_files, get_dipole_info
from emoles.constant import atom_to_transform_indices, convention_dict
from emoles.utils import (
    cut_and_cal_matrix,
    format_number,
    vec_cosine_similarity,
    get_shifted_ham,
    get_mo_occ,
    matrix_transform,
    generate_molecule_transform_indices,
)
from emoles.inference.common_tools import calculate_esp_from_dm


def calculate_dm_dipole_mae(pred_dm, target_dm, mol):
    """
    Calculate density matrix and dipole errors.
    """
    error_dict = {}
    # Density matrix MAE
    diff_matrix = np.abs(np.array(pred_dm - target_dm))
    error_dict["density_matrix"] = np.mean(diff_matrix)
    # Dipole moment error
    dip_pred = get_dipole_info(mol, pred_dm)
    dip_target = get_dipole_info(mol, target_dm)
    error_dict["dipole"] = np.abs(np.array(dip_pred - dip_target))
    return error_dict


def calculate_properties_from_dm(mol, dm, prefix, gen_dm_flag=False):
    """
    A helper function to:
    1. Generate an fchk file from a density matrix via mokit.
    2. Compute ESP properties using Multiwfn.
    3. Compute Deformation Factor (phi) for the first Li atom found.

    Parameters:
    -----------
    mol : gto.Mole
        PySCF molecule object.
    dm : np.ndarray
        Density matrix.
    prefix : str
        A prefix for all output files (e.g., "pred", "target").
        Also used to name the Li PDB file (e.g. Li_pred_1_...).

    Returns:
    --------
    tuple
        (ESP_max_in_eV, ESP_min_in_eV, deformation_factor)
        deformation_factor is a float (phi) or None if calculation fails/no Li.
    """
    import json
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft
    from emoles.multiwfn import ESPCalculator, ELFDeformationCalculator

    # 1. Set up and run PySCF DFT calculation from the density matrix
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    fock = mf.get_fock(dm=dm)
    s = mf.get_ovlp()
    orbital_energies, orbital_coefficients = mf.eig(fock, s)
    mf.mo_energy = orbital_energies
    mf.mo_coeff = orbital_coefficients
    mf.dm = dm

    # 2. Generate fchk file
    fch_filename = f"{prefix}.fch"
    fchk(mf, fch_filename, density=True)

    # --- Part A: ESP Calculation ---
    esp_calculator = ESPCalculator(fch_filename)
    esp_results = esp_calculator.get_ESP_value()

    if gen_dm_flag:
        esp_calculator.get_acc_grid_data()

    # Save ESP info
    with open(f"{prefix}_esp_info.json", "w") as f:
        json.dump(esp_results, fp=f, indent=4)

    esp_max = esp_results.get("ESP_max_eV", 0)
    esp_min = esp_results.get("ESP_min_eV", 0)

    # --- Part B: Li Deformation Factor Calculation ---
    li_phi = None

    # Get symbols and coordinates (ensure Angstroms for geometry analysis)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords_ang = mol.atom_coords(unit="Ang")

    # Find Li atoms
    li_indices_0based = [i for i, s in enumerate(symbols) if s == "Li"]

    if li_indices_0based:
        # Default: Only handle the first Li atom found
        target_li_idx = li_indices_0based[0]
        i_1based = target_li_idx + 1
        li_center = coords_ang[target_li_idx]

        # Initialize ELF Calculator
        # Using parameters consistent with your main script
        elf_calculator = ELFDeformationCalculator(
            fch_filename, isovalue=0.5, diff_list=[0.09], li_cutoff=1.1
        )

        # Determine ID name for PDB saving (e.g., "pred_1")
        # Passing a value to li_id triggers PDB saving in your ELFDeformationCalculator
        save_id = f"{prefix}_{i_1based}"

        try:
            elf_res = elf_calculator.calculate(
                atom_index_1based=i_1based,
                li_center=li_center,
                li_id=save_id,  # This ensures PDB is saved
                radius=3.0,
                grid_spacing=0.1,
            )

            # Extract phi for the specific diff (0.09)
            target_diff = 0.09
            if elf_res and target_diff in elf_res:
                li_phi = elf_res[target_diff]["phi"]
        except Exception as e:
            print(f"Warning: Failed to calculate phi for {prefix} Li: {e}")

    return esp_max, esp_min, li_phi


def process_loss_dict(data, item_flag=False, key="pred_vs_label"):
    if key:
        data = data[key]

    # Convert and format values
    processed = {
        "Density-Matrix": data["density_matrix"],
        "Dipole-Moment-magnitude": data["dipole"],
        # Convert to 1e-6 Ha
        "Ham-MAE (1e-6 Ha)": data["hamiltonian"] * 1e6,
        "Diag-MAE (1e-6 Ha)": data["diagonal_hamiltonian_mae"] * 1e6,
        "NonDiag-MAE (1e-6 Ha)": data["non_diagonal_hamiltonian_mae"] * 1e6,
        "Shifted-Ham-MAE (1e-6 Ha)": data["shifted_ham"] * 1e6,
        "Shifted-Diag-MAE (1e-6 Ha)": data["shifted_diagonal_hamiltonian_mae"] * 1e6,
        "Shifted-NonDiag-MAE (1e-6 Ha)": data["shifted_non_diagonal_hamiltonian_mae"] * 1e6,
        "occ-orb-MAE (1e-6 Ha)": data["occupied_orbital_energy"] * 1e6,
        # Convert to percentage
        "occ-orb-Sim (%)": data["orbital_coefficients"] * 1e2,
        "HOMO-Sim (%)": data["HOMO_coefficients"] * 1e2,
        "LUMO-Sim (%)": data["LUMO_coefficients"] * 1e2,
        # Keep in eV
        "HOMO (eV)": data["HOMO"],
        "LUMO (eV)": data["LUMO"],
        "GAP (eV)": data["GAP"],
    }
    if item_flag:
        processed["Time (s/item)"] = data["second_per_item"]
        processed["Total Items"] = int(data["total_items"])
    # Format numbers according to the rules
    for key_, value in processed.items():
        if key_ != "Total Items":  # Skip integer values
            processed[key_] = float(format_number(value))

    return processed


def criterion(outputs, target, names, flag=None, atoms=None, mol=None):
    error_dict = {}
    for key in names:
        if key == "orbital_coefficients":
            output_orbital_coefficients, target_orbital_coefficients = (
                torch.from_numpy(outputs[key]).T,
                torch.from_numpy(target[key]).T,
            )
            aa = torch.cosine_similarity(
                output_orbital_coefficients, target_orbital_coefficients
            ).abs().numpy()
            error_dict[key] = (
                torch.cosine_similarity(
                    output_orbital_coefficients, target_orbital_coefficients
                )
                .abs()
                .mean()
                .numpy()
            )
            if flag:
                print(output_orbital_coefficients.shape)
                print(target_orbital_coefficients.shape)
                print(aa)
                print(error_dict[key])
        elif key in ["LUMO_coefficients", "HOMO_coefficients"]:
            error_dict[key] = vec_cosine_similarity(outputs[key], target[key])
            if flag:
                print(error_dict[key])
        elif key == "density_matrix":
            dm_output = make_rdm1(mo_coeff=outputs[key], mo_occ=outputs["mo_occ"])
            dm_target = make_rdm1(mo_coeff=target[key], mo_occ=outputs["mo_occ"])
            if mol:
                dip_output = get_dipole_info(mol, dm_output)
                dip_target = get_dipole_info(mol, dm_target)
                error_dict["dipole"] = np.abs(np.array(dip_output - dip_target))
            diff_matrix = np.abs(np.array(dm_output - dm_target))
            error_dict[key] = np.mean(diff_matrix)
        elif key == "shifted_ham":
            diff_matrix = np.abs(np.array(outputs[key] - target[key]))
            error_dict[key] = np.mean(diff_matrix)

            if atoms:
                _, atom_in_mo_indices = generate_molecule_transform_indices(
                    atom_types=atoms.symbols,
                    atom_to_transform_indices=atom_to_transform_indices,
                )
                (
                    error_dict["shifted_diagonal_hamiltonian_mae"],
                    error_dict["shifted_non_diagonal_hamiltonian_mae"],
                ) = cut_and_cal_matrix(
                    full_matrix=diff_matrix, atom_in_mo_indices=atom_in_mo_indices
                )
        elif key == "hamiltonian":
            diff_matrix = np.abs(np.array(outputs[key] - target[key]))
            error_dict[key] = np.mean(diff_matrix)
            if atoms:
                _, atom_in_mo_indices = generate_molecule_transform_indices(
                    atom_types=atoms.symbols,
                    atom_to_transform_indices=atom_to_transform_indices,
                )
                (
                    error_dict["diagonal_hamiltonian_mae"],
                    error_dict["non_diagonal_hamiltonian_mae"],
                ) = cut_and_cal_matrix(
                    full_matrix=diff_matrix[0],
                    atom_in_mo_indices=atom_in_mo_indices,
                )
        else:
            diff = np.array(outputs[key] - target[key])
            mae = np.mean(np.abs(diff))

            if key in ["HOMO", "LUMO", "GAP"]:
                mae = mae * Hartree

            error_dict[key] = mae
            if flag:
                print(key)
                print(error_dict[key])
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(
        np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian),
        frac_overlap,
    )
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def post_processing(batch, default_type=np.float32):
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray) and np.issubdtype(
                batch[key].dtype, np.floating
        ):
            batch[key] = batch[key].astype(default_type)
    return batch


def load_gaussian_data(idx, gau_npy_folder_path, united_overlap_flag):
    gau_path = os.path.join(gau_npy_folder_path, f"{idx}")
    gau_ham = np.load(os.path.join(gau_path, "fock.npy"))
    # gau_ham = np.load(os.path.join(gau_path, 'original_ham.npy'))[0]
    if not united_overlap_flag:
        gau_overlap = np.load(os.path.join(gau_path, "overlap.npy"))
        return gau_ham, gau_overlap
    return gau_ham, None


def process_dm_loss_dict(data, key="pred_vs_label"):
    """Process and format the loss dictionary for density matrix evaluation"""
    if key and key in data:
        data = data[key]

    processed = {}

    # --- 1. 基础 DM & Dipole 指标 ---
    if "density_matrix" in data:
        processed["Density-Matrix-MAE"] = data["density_matrix"]
    if "diagonal_density_matrix_mae" in data:
        processed["Diag-DM-MAE"] = data["diagonal_density_matrix_mae"]
    if "non_diagonal_density_matrix_mae" in data:
        processed["NonDiag-DM-MAE"] = data["non_diagonal_density_matrix_mae"]

    if "dipole" in data:
        processed["Dipole-Moment-MAE-Debye"] = data["dipole"]

    # --- 1b. 新增: 电子数守恒误差 ---
    if "pred_electron_number_error" in data:
        processed["Pred-Ne-Error"] = data["pred_electron_number_error"]
    if "target_electron_number_error" in data:
        processed["Label-Ne-Error"] = data["target_electron_number_error"]
    if "electron_number_pred_vs_target_error" in data:
        processed["Ne-Pred-vs-Label-Error"] = data[
            "electron_number_pred_vs_target_error"
        ]

    # --- 2. 新增: Ham, 能级与轨道指标 (从 DM 推导) ---
    # Hamiltonian MAE
    if "hamiltonian" in data:
        processed["Ham-MAE (1e-6 Ha)"] = data["hamiltonian"] * 1e6
    if "diagonal_hamiltonian_mae" in data:
        processed["Diag-Ham-MAE (1e-6 Ha)"] = data["diagonal_hamiltonian_mae"] * 1e6
    if "non_diagonal_hamiltonian_mae" in data:
        processed["NonDiag-Ham-MAE (1e-6 Ha)"] = data["non_diagonal_hamiltonian_mae"] * 1e6

    # Orbital Energies (HOMO/LUMO/GAP)
    if "HOMO" in data:
        processed["HOMO (eV)"] = data["HOMO"]
    if "LUMO" in data:
        processed["LUMO (eV)"] = data["LUMO"]
    if "GAP" in data:
        processed["GAP (eV)"] = data["GAP"]

    # Orbital Similarities
    if "HOMO_coefficients" in data:
        processed["HOMO-Sim (%)"] = data["HOMO_coefficients"] * 1e2
    if "LUMO_coefficients" in data:
        processed["LUMO-Sim (%)"] = data["LUMO_coefficients"] * 1e2
    if "orbital_coefficients" in data:
        processed["occ-orb-Sim (%)"] = data["orbital_coefficients"] * 1e2

    # --- 3. ESP 指标 ---
    if "esp_max_mae" in data:
        processed["ESP-Max-MAE-eV"] = data["esp_max_mae"]
    elif "esp_max" in data:
        processed["ESP-Max-MAE-eV"] = data["esp_max"]

    if "esp_min_mae" in data:
        processed["ESP-Min-MAE-eV"] = data["esp_min_mae"]
    elif "esp_min" in data:
        processed["ESP-Min-MAE-eV"] = data["esp_min"]

    # --- 4. Deformation Factor ---
    if "deformation_factor_mae" in data:
        processed["Deformation Factor MAE"] = data["deformation_factor_mae"]

    # --- 5. 统计信息 ---
    if "second_per_item" in data:
        processed["Time (s/item)"] = data["second_per_item"]

    # 格式化数字
    try:
        for k, v in processed.items():
            if k not in ["Total Items", "Attempted Items", "Failed Items"]:
                processed[k] = float(format_number(v))
    except NameError:
        pass

    return processed


def get_electronic_properties(
        mol, ham=None, overlap=None, dm=None, shifted_ham=None
):
    """
    Helper function to extract electronic properties (Energies, Orbitals, Gap)
    from either Hamiltonian+Overlap OR Density Matrix.
    """
    # 1. 准备 Hamiltonian 和 Overlap
    if ham is None:
        if dm is None:
            raise ValueError("Must provide either Hamiltonian or Density Matrix")
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        # PySCF get_fock returns (N, N) for RKS usually, but let's handle potential (1, N, N)
        ham = mf.get_fock(dm=dm)
        if overlap is None:
            overlap = mf.get_ovlp()

    if overlap is None:
        overlap = mol.intor("int1e_ovlp")

    # 2. 规范化 Ham 和 Overlap 维度
    # 先确保它们是 2D (N, N) 用于计算
    ham_2d = ham
    if ham.ndim == 3:
        ham_2d = ham[0]  # Assume (1, N, N) -> (N, N)

    ov_2d = overlap
    if overlap.ndim == 3:
        ov_2d = overlap[0]

    # 3. 求解广义特征值问题 (使用 3D 输入适配函数)
    # cal_orbital_and_energies 需要 (Batch, N, N) 输入
    ham_in_3d = ham_2d[None, ...]  # Expand to (1, N, N)
    ov_in_3d = ov_2d[None, ...]  # Expand to (1, N, N)

    energies, coeffs = cal_orbital_and_energies(
        overlap_matrix=ov_in_3d, full_hamiltonian=ham_in_3d
    )

    # 4. 确定占据数和索引
    n_electrons = mol.tot_electrons()
    homo_idx = int(n_electrons / 2) - 1
    mo_occ = get_mo_occ(full_len=len(energies), occ_len=homo_idx + 1)

    # 5. 处理 Shifted Ham (Criterion 期望 shifted_ham 是 2D (N, N))
    if shifted_ham is None:
        shifted_ham_2d = ham_2d
    else:
        shifted_ham_2d = shifted_ham
        if shifted_ham_2d.ndim == 3:
            shifted_ham_2d = shifted_ham_2d[0]

    # 6. 打包结果
    # 关键修正：Criterion 中的 'hamiltonian' 分支会执行 diff_matrix[0]，所以这里必须给 3D (1, N, N)
    results = {
        "HOMO": energies[homo_idx],
        "LUMO": energies[homo_idx + 1],
        "GAP": energies[homo_idx + 1] - energies[homo_idx],
        "hamiltonian": ham_in_3d,  # (1, N, N)
        "overlap": ov_in_3d,  # (1, N, N)
        "shifted_ham": shifted_ham_2d,  # (N, N)
        "density_matrix": dm
        if dm is not None
        else make_rdm1(mo_coeff=coeffs, mo_occ=mo_occ),
        "mo_occ": mo_occ,
        "orbital_coefficients": coeffs[:, : homo_idx + 1],
        "HOMO_coefficients": coeffs[:, homo_idx],
        "LUMO_coefficients": coeffs[:, homo_idx + 1],
        "occupied_orbital_energy": energies[: homo_idx + 1],
    }
    return results


def get_electron_number_from_dm(dm, overlap):
    """
    根据 AO 密度矩阵 P 和 AO 重叠矩阵 S 计算电子数:
        N_e = Tr(P S)

    支持:
    - dm: (nao, nao) 或 (spin, nao, nao)
    - overlap: (nao, nao) 或 (1, nao, nao)
    """
    P = dm
    S = overlap

    # 如果是自旋分辨的 3D DM，则先对自旋求和
    if P.ndim == 3:
        P = P.sum(axis=0)

    if S.ndim == 3:
        S = S[0]

    return float(np.einsum("ij,ji->", P, S))


def evaluate_dm_from_npy(
        abs_ase_path,
        npy_folder_path,
        convention="def2svp",
        mol_charge=0,
        pred_dm_filename="predicted_dm.npy",
        target_dm_filename="target_dm.npy",
        transform_dm_flag=True,
        get_esp_sta_flag=True,
        get_ham_flag=True,
        keep_xyz_file=True,
        n_save_cube_items: int = 5,  # 新增：专门用于 generate_cube_files 的保存数量
        temp_data_file: str = "temp_cube_data.pkl",  # 新增：保存路径
        max_items: int = 300,
        gen_esp_cube_flag: bool = False,
        summary_filename="evaluation_summary.npz",
):
    import time
    import json
    import numpy as np
    import traceback
    import pickle  # 需要 pickle
    from ase.db import connect
    from ase.io import write
    from tqdm import tqdm
    import pyscf
    from pyscf import tools

    def format_number_local(x):
        return "{:.6f}".format(x)

    if convention == "6311gdp":
        basis = "6-311+g(d,p)"
        back_convention = "back_2_thu_pyscf"
    elif convention == "back_thu_cluster":
        basis = "def2svp"
        back_convention = convention
    else:
        basis = "def2svp"
        back_convention = "back2pyscf"

    total_error_dict = {"total_items": 0, "pred_vs_label": {}}
    start_time = time.time()
    fail_count = 0
    attempted_count = 0
    failed_indices = []
    summary_data_list = []

    # 用于 generate_cube_files 的临时数据列表
    temp_cube_data = []

    def _load_npy_safe(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            if idx == max_items:
                break
            attempted_count += 1
            cwd_ = os.getcwd()
            try:
                work_dir = os.path.join(npy_folder_path, f"{idx}")
                if not os.path.exists(work_dir):
                    continue
                os.chdir(work_dir)

                atom_nums = a_row.numbers
                an_atoms = a_row.toatoms()

                pred_dm = _load_npy_safe(pred_dm_filename)
                target_dm = _load_npy_safe(target_dm_filename)

                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)
                    target_dm = matrix_transform(target_dm, atom_nums, convention=back_convention)

                current_mol_charge = a_row.data.get("charge", mol_charge)

                print('===============')
                print('===============')
                print(current_mol_charge)
                print('===============')
                print('===============')

                sum_of_atomic_numbers = an_atoms.get_atomic_numbers().sum()
                total_electrons = sum_of_atomic_numbers - current_mol_charge
                mol_spin = total_electrons % 2

                mol = pyscf.gto.Mole()
                t = [
                    [atom_nums[atom_idx], an_atom.position]
                    for atom_idx, an_atom in enumerate(an_atoms)
                ]
                mol.charge = current_mol_charge
                mol.spin = mol_spin
                mol.build(verbose=0, atom=t, basis=basis, unit="ang")
                overlap = mol.intor("int1e_ovlp")

                # 计算由 DM 与 S 得到的电子数
                expected_electrons = float(total_electrons)
                Ne_pred = get_electron_number_from_dm(pred_dm, overlap)
                Ne_target = get_electron_number_from_dm(target_dm, overlap)

                # 1. 基础误差 (Density Matrix & Dipole)
                errors = calculate_dm_dipole_mae(pred_dm, target_dm, mol)

                # --- NEW: Calculate Block Diagonal/Non-Diagonal MAE for Density Matrix ---
                _, atom_in_mo_indices = generate_molecule_transform_indices(
                    atom_types=an_atoms.get_chemical_symbols(),
                    atom_to_transform_indices=atom_to_transform_indices,
                )
                # Recalculate diff locally to split it
                dm_diff = np.abs(pred_dm - target_dm)
                dm_diag, dm_non_diag = cut_and_cal_matrix(
                    full_matrix=dm_diff,
                    atom_in_mo_indices=atom_in_mo_indices
                )
                errors["diagonal_density_matrix_mae"] = dm_diag
                errors["non_diagonal_density_matrix_mae"] = dm_non_diag
                # -----------------------------------------------------------------------

                # 1b. 新增：电子数守恒相关误差
                errors["pred_electron_number_error"] = abs(Ne_pred - expected_electrons)
                errors["target_electron_number_error"] = abs(
                    Ne_target - expected_electrons
                )
                errors["electron_number_pred_vs_target_error"] = abs(
                    Ne_pred - Ne_target
                )

                # 2. (可选) 计算 DM 推导出的 Hamiltonian 误差及轨道相似度
                if get_ham_flag:
                    pred_props = get_electronic_properties(
                        mol, dm=pred_dm, overlap=overlap
                    )
                    target_props = get_electronic_properties(
                        mol, dm=target_dm, overlap=overlap
                    )

                    eval_keys = [
                        "hamiltonian",
                        "HOMO",
                        "LUMO",
                        "GAP",
                        "orbital_coefficients",
                        "HOMO_coefficients",
                        "LUMO_coefficients",
                    ]

                    # Note: criterion will handle diagonal/non-diagonal for hamiltonian
                    # because we pass 'atoms=an_atoms' and key 'hamiltonian' is in eval_keys.
                    ham_orb_errors = criterion(
                        pred_props,
                        target_props,
                        eval_keys,
                        flag=False,
                        atoms=an_atoms,
                        mol=mol,
                    )
                    errors.update(ham_orb_errors)

                    # --- 保存用于 generate_cube_files 的数据 ---
                    if idx < n_save_cube_items:
                        # 构建 mol_info 用于重建 PySCF Mole
                        mol_info = {
                            "atom_nums": [int(x) for x in atom_nums],
                            "atom_coords": [list(at.position) for at in an_atoms],
                            "charge": int(current_mol_charge),
                            "spin": int(mol_spin),
                            "basis": basis,
                            "unit": "ang",
                        }

                        # 构建单个 item 数据
                        cube_item = {
                            "idx": idx,
                            "HOMO_sim": errors.get("HOMO_coefficients", 0.0),
                            "mol_info": mol_info,
                            "outputs": pred_props,
                            "tgt_info": target_props,
                        }
                        temp_cube_data.append(cube_item)
                    # ----------------------------------------------------

                print(errors)

                # 3. 计算 ESP 和 Deformation Factor
                if get_esp_sta_flag:
                    p_esp_max, p_esp_min, p_phi = calculate_properties_from_dm(
                        mol, pred_dm, "pred", gen_dm_flag=gen_esp_cube_flag
                    )
                    t_esp_max, t_esp_min, t_phi = calculate_properties_from_dm(
                        mol, target_dm, "target", gen_dm_flag=gen_esp_cube_flag
                    )
                    errors["esp_max_mae"] = abs(t_esp_max - p_esp_max)
                    errors["esp_min_mae"] = abs(t_esp_min - p_esp_min)
                    if p_phi is not None and t_phi is not None:
                        errors["deformation_factor_mae"] = abs(p_phi - t_phi)
                    else:
                        errors["deformation_factor_mae"] = None

                if keep_xyz_file:
                    write("atomic_structure.xyz", an_atoms)

                for key, val in errors.items():
                    if val is not None:
                        total_error_dict["pred_vs_label"][key] = (
                                total_error_dict["pred_vs_label"].get(key, 0.0) + val
                        )

                total_error_dict["total_items"] += 1

                mol_info_log = {
                    "formula": an_atoms.get_chemical_formula(),
                    "charge": int(current_mol_charge),
                    "spin": int(mol_spin),
                }
                local_result = {"idx": idx, "mol_info": mol_info_log, "errors": errors}
                with open("dm_evaluation_result.json", "w") as f_json:
                    json.dump(local_result, f_json, indent=4, default=str)

                flat_data = {"idx": idx}
                flat_data.update(errors)
                summary_data_list.append(flat_data)

            except Exception as e:
                fail_count += 1
                failed_indices.append(idx)
                traceback.print_exc()
                print(f"[evaluate_dm_from_npy] idx {idx} failed: {repr(e)}")
            finally:
                os.chdir(cwd_)

    # 循环结束后，保存 temp_cube_data
    if temp_data_file and len(temp_cube_data) > 0:
        save_path = os.path.join(npy_folder_path, temp_data_file)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(temp_cube_data, f)
            print(
                f"[evaluate_dm_from_npy] Saved cube info for {len(temp_cube_data)} items to {save_path}"
            )
        except Exception as e:
            print(f"[evaluate_dm_from_npy] Failed to save temp cube data: {e}")

    n = total_error_dict["total_items"]
    if n > 0:
        for key in total_error_dict["pred_vs_label"].keys():
            total_error_dict["pred_vs_label"][key] /= n

    end_time = time.time()
    total_error_dict["second_per_item"] = (end_time - start_time) / max(1, n)

    if len(summary_data_list) > 0:
        all_keys = set().union(*(d.keys() for d in summary_data_list))
        npz_dict = {}
        for key in all_keys:
            values = []
            for item in summary_data_list:
                val = item.get(key, np.nan)
                if val is None:
                    val = np.nan
                values.append(val)
            npz_dict[key] = np.array(values)
        np.savez(os.path.join(npy_folder_path, summary_filename), **npz_dict)

    final_data_to_process = {
        "pred_vs_label": total_error_dict["pred_vs_label"].copy()
    }
    final_data_to_process["pred_vs_label"]["second_per_item"] = total_error_dict[
        "second_per_item"
    ]

    result_dict = process_dm_loss_dict(final_data_to_process, key="pred_vs_label")
    result_dict["Total Items"] = n
    result_dict["Attempted Items"] = attempted_count
    result_dict["Failed Items"] = fail_count

    print(
        f"[evaluate_dm_from_npy] Attempted: {attempted_count}, Success: {n}, Failed: {fail_count}"
    )
    return result_dict


def prepare_np(
        overlap_matrix,
        full_hamiltonian,
        atom_symbols,
        transform_ham_flag=False,
        transform_overlap_flag=False,
        convention="def2svp",
):
    if convention == "6311gdp":
        back_convention = "back_2_thu_pyscf"
    else:
        back_convention = "back2pyscf"

    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(
            full_hamiltonian, atom_symbols, convention=back_convention
        )
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(
            overlap_matrix, atom_symbols, convention=back_convention
        )
    return full_hamiltonian, overlap_matrix


def get_mae_from_npy(
        abs_ase_path,
        npy_folder_path,
        temp_data_file=None,
        united_overlap_flag=False,
        convention="def2svp",
        mol_charge=0,
        save_summary=False,
        full_save_items=10,
):
    import pickle

    if convention == "6311gdp":
        basis = "6-311+g(d,p)"
        back_convention = "back_2_thu_pyscf"
    else:
        basis = "def2svp"
        back_convention = "back2pyscf"

    total_error_dict = {"total_items": 0, "pred_vs_label": {}}
    start_time = time.time()
    temp_data = []

    def _load_npy_safe(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            atom_nums = a_row.numbers
            an_atoms = a_row.toatoms()
            total_error_dict["total_items"] += 1

            pred_ham = _load_npy_safe(
                os.path.join(npy_folder_path, f"{idx}", "predicted_ham.npy")
            )
            orig_ham = _load_npy_safe(
                os.path.join(npy_folder_path, f"{idx}", "original_ham.npy")
            )

            mol = pyscf.gto.Mole()
            t = [
                [atom_nums[atom_idx], an_atom.position]
                for atom_idx, an_atom in enumerate(an_atoms)
            ]
            mol_charge = a_row.data.get("charge", mol_charge)
            mol.charge = mol_charge
            sum_of_atomic_numbers = an_atoms.get_atomic_numbers().sum()
            total_electrons = sum_of_atomic_numbers - mol_charge
            mol_spin = total_electrons % 2
            mol.spin = mol_spin
            mol.build(verbose=0, atom=t, basis=basis, unit="ang")

            # 准备矩阵数据
            shifted_label_ham = None  # 默认 None

            if not united_overlap_flag:
                pred_ov = _load_npy_safe(
                    os.path.join(npy_folder_path, f"{idx}", "predicted_overlap.npy")
                )
                orig_ov = _load_npy_safe(
                    os.path.join(npy_folder_path, f"{idx}", "original_overlap.npy")
                )

                orig_ham_prep, orig_ov_prep = prepare_np(
                    atom_symbols=atom_nums,
                    overlap_matrix=orig_ov,
                    full_hamiltonian=orig_ham,
                    transform_ham_flag=True,
                    transform_overlap_flag=True,
                    convention=convention,
                )
                pred_ham_prep, pred_ov_prep = prepare_np(
                    atom_symbols=atom_nums,
                    overlap_matrix=pred_ov,
                    full_hamiltonian=pred_ham,
                    transform_ham_flag=True,
                    transform_overlap_flag=True,
                    convention=convention,
                )
            else:
                orig_ham_bt = matrix_transform(
                    orig_ham, atom_nums, convention=back_convention
                )
                pred_ham_bt = matrix_transform(
                    pred_ham, atom_nums, convention=back_convention
                )
                target_overlap = mol.intor("int1e_ovlp")

                # 计算 Shifted Ham (仅针对 Target)
                shifted_label_ham = get_shifted_ham(
                    predicted_ham=pred_ham_bt,
                    label_ham=orig_ham_bt,
                    overlap=target_overlap,
                )

                orig_ham_prep, orig_ov_prep = prepare_np(
                    atom_symbols=atom_nums,
                    overlap_matrix=target_overlap,
                    full_hamiltonian=orig_ham_bt,
                    transform_ham_flag=False,
                    transform_overlap_flag=False,
                    convention=convention,
                )
                pred_ham_prep, pred_ov_prep = prepare_np(
                    atom_symbols=atom_nums,
                    overlap_matrix=target_overlap,
                    full_hamiltonian=pred_ham_bt,
                    transform_ham_flag=False,
                    transform_overlap_flag=False,
                    convention=convention,
                )

            # --- 使用 Helper 函数获取属性 ---
            # prepare_np 返回的是 (1, N, N) 的 3D 数组，Helper 会处理
            outputs = get_electronic_properties(
                mol, ham=pred_ham_prep, overlap=pred_ov_prep
            )
            tgt_info = get_electronic_properties(
                mol,
                ham=orig_ham_prep,
                overlap=orig_ov_prep,
                shifted_ham=shifted_label_ham,
            )

            # 计算误差
            pred_vs_label = criterion(
                outputs, tgt_info, list(outputs.keys()), flag=False, atoms=an_atoms, mol=mol
            )
            for key, val in pred_vs_label.items():
                total_error_dict["pred_vs_label"][key] = (
                        total_error_dict["pred_vs_label"].get(key, 0.0) + val
                )

            if save_summary:
                mol_info = {
                    "atom_nums": [int(x) for x in atom_nums],
                    "atom_coords": [list(at.position) for at in an_atoms],
                    "charge": mol_charge,
                    "spin": mol_spin,
                    "basis": basis,
                    "unit": "ang",
                }
                item_data = {
                    "mol_info": mol_info,
                    "pred_vs_label": pred_vs_label,
                    "HOMO_sim": pred_vs_label["HOMO_coefficients"],
                    "idx": idx,
                }
                if idx < full_save_items:
                    item_data.update({"outputs": outputs, "tgt_info": tgt_info})
                temp_data.append(item_data)

    n = total_error_dict["total_items"]
    if n > 0:
        for key in total_error_dict["pred_vs_label"].keys():
            total_error_dict["pred_vs_label"][key] = (
                    total_error_dict["pred_vs_label"][key] / n
            )

    end_time = time.time()
    total_error_dict["second_per_item"] = (end_time - start_time) / max(1, n)

    total_error_dict = process_loss_dict(total_error_dict, key="pred_vs_label")

    print(total_error_dict)

    if save_summary and temp_data_file is not None:
        with open(temp_data_file, "wb") as f:
            pickle.dump(temp_data, f)

    return total_error_dict



from itertools import permutations
from types import SimpleNamespace


def find_best_dm_transform_permutation(
        abs_ase_path,
        npy_folder_path,
        dm_filename="predicted_dm.npy",
        basis_set="def2svp",
        n_test_items=5,
        base_convention="back2pyscf"  # 作为模板的基础配置，包含 atom_to_orbitals_map 等
):
    """
    轻量级测试函数：遍历 P 和 D 轨道的索引排列，寻找使电子数误差 (Tr(PS) - Ne) 最小的变换方案。
    """
    import numpy as np
    from pyscf import gto
    from ase.db import connect
    from tqdm import tqdm

    # 1. 准备排列组合
    p_perms = list(permutations([0, 1, 2]))
    d_perms = list(permutations([0, 1, 2, 3, 4]))

    # 存储每种 (p_idx, d_idx) 组合的累计误差
    # key: (tuple_p, tuple_d), value: cumulative_error
    permutation_errors = {}

    # 初始化所有组合的误差为 0
    for p in p_perms:
        for d in d_perms:
            permutation_errors[(p, d)] = 0.0

    print(f"--- Starting Permutation Search ---")
    print(f"Testing {len(p_perms) * len(d_perms)} combinations on {n_test_items} molecules...")

    # 2. 遍历分子
    valid_items = 0
    with connect(abs_ase_path) as db:
        for idx, row in tqdm(enumerate(db.select()), total=n_test_items):
            if valid_items >= n_test_items:
                break

            # 加载 DM
            folder = os.path.join(npy_folder_path, str(idx))
            dm_path = os.path.join(folder, dm_filename)
            if not os.path.exists(dm_path):
                continue

            try:
                # 原始数据
                orig_dm = np.load(dm_path)
                atom_nums = row.numbers
                coords = row.toatoms().positions
                charge = row.data.get("charge", 0)

                # 构建 PySCF 分子以获取重叠矩阵 S
                mol = gto.M(
                    atom=[(atom_nums[i], coords[i]) for i in range(len(atom_nums))],
                    basis=basis_set,
                    charge=charge,
                    spin=(sum(atom_nums) - charge) % 2,
                    unit='Ang',
                    verbose=0
                )
                overlap = mol.intor("int1e_ovlp")
                target_ne = float(mol.nelectron)

                # 3. 核心循环：测试每种排列
                # 我们需要临时修改全局 convention_dict 或传入一个新的 key
                # 这里我们利用 base_convention 作为模板
                template_conf = convention_dict[base_convention]

                # 临时 key
                temp_key = "temp_perm_search"

                for p_idx in p_perms:
                    for d_idx in d_perms:
                        # 构造新的 Namespace 配置
                        new_conf = SimpleNamespace(
                            atom_to_orbitals_map=template_conf.atom_to_orbitals_map,
                            orbital_sign_map=template_conf.orbital_sign_map,
                            orbital_order_map=template_conf.orbital_order_map,
                            # 关键：在这里应用当前的排列
                            orbital_idx_map={
                                's': [0],
                                'p': list(p_idx),
                                'd': list(d_idx)
                            }
                        )

                        # 注入全局字典 (matrix_transform 依赖全局 convention_dict)
                        convention_dict[temp_key] = new_conf

                        # 执行变换
                        transformed_dm = matrix_transform(orig_dm, atom_nums, convention=temp_key)

                        # 计算电子数 Ne = Tr(P @ S)
                        # 注意 transformed_dm 可能是 (1, N, N) 或 (N, N)
                        if transformed_dm.ndim == 3:
                            dm_2d = transformed_dm[
                                0]  # RKS/spin-sum usually done externally, but taking first dim if structure matches
                            # 如果是 spin separated (2, N, N), 应该 sum(axis=0)，视你的数据格式而定
                            # 假设输入是 RKS 或已处理过的 DM
                            if transformed_dm.shape[0] == 2:
                                dm_2d = np.sum(transformed_dm, axis=0)
                        else:
                            dm_2d = transformed_dm

                        ne_calc = np.einsum('ij,ji->', dm_2d, overlap)

                        # 累加绝对误差
                        error = abs(ne_calc - target_ne)
                        permutation_errors[(p_idx, d_idx)] += error

                valid_items += 1

            except Exception as e:
                print(f"Skipping idx {idx} due to error: {e}")
                continue

    # 4. 寻找最佳结果
    if valid_items == 0:
        print("No valid items processed.")
        return

    best_combo = min(permutation_errors, key=permutation_errors.get)
    best_p, best_d = best_combo
    min_error = permutation_errors[best_combo] / valid_items

    print("\n" + "=" * 50)
    print(f"  SEARCH COMPLETE")
    print("=" * 50)
    print(f"Best Avg Electron Error: {min_error:.2e}")
    print(f"Best P-permutation: {list(best_p)}")
    print(f"Best D-permutation: {list(best_d)}")
    print("-" * 50)
    print("Suggested Convention Dict Entry:\n")

    # 格式化输出 Python 代码
    print(f"'best_found_convention': Namespace(")
    print(f"    atom_to_orbitals_map={template_conf.atom_to_orbitals_map},")
    print(f"    orbital_idx_map={{'s': [0], 'p': {list(best_p)}, 'd': {list(best_d)}}},")
    print(f"    orbital_sign_map={template_conf.orbital_sign_map},")
    print(f"    orbital_order_map={template_conf.orbital_order_map}")
    print(f"),")
    print("=" * 50)

    # 清理临时 key
    if "temp_perm_search" in convention_dict:
        del convention_dict["temp_perm_search"]

# 使用示例 (请根据实际路径调用):
# find_best_dm_transform_permutation(
#     abs_ase_path="path/to/test.db",
#     npy_folder_path="path/to/npy",
#     basis_set="def2svp",  # 或 "6-311+g(d,p)"
#     base_convention="back2pyscf" # 选择一个原子轨道结构(ssp vs sssp)与你目标一致的作为模板
# )