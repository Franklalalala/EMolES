import os
import time
import pickle
import json


os.environ['PYSCF_MAX_MEMORY'] = '32000'

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
from pyscf.data import radii

# ============================================================
# UFF VDW 半径表 (Å) - 对齐 Gaussian Radii=UFF
# ============================================================
UFF_RADII_ANG = {
    1: 1.4430,   2: 1.1810,   3: 1.2255,   4: 1.3725,   5: 1.8150,
    6: 1.9255,   7: 1.8300,   8: 1.7500,   9: 1.6820,  10: 1.6215,
   11: 1.4915,  12: 1.5105,  13: 2.2495,  14: 2.1475,  15: 2.0735,
   16: 2.0175,  17: 2.0450,  18: 1.9340,  19: 1.9060,  20: 1.6995,
   21: 1.6475,  22: 1.5875,  23: 1.5720,  24: 1.5115,  25: 1.4805,
   26: 1.4560,  27: 1.4360,  28: 1.4170,  29: 1.7475,  30: 1.3815,
   31: 2.1915,  32: 2.1400,  33: 2.1150,  34: 2.1025,  35: 2.1650,
   36: 2.0200,  37: 2.2585,  38: 2.0515,  39: 1.8245,  40: 1.6155,
   41: 1.5720,  42: 1.5260,  43: 1.4990,  44: 1.4815,  45: 1.4645,
   46: 1.4495,  47: 1.5740,  48: 1.4240,  49: 2.2315,  50: 2.1960,
   51: 2.2100,  52: 2.2350,  53: 2.3600,  54: 2.1815,
}

BOHR = radii.BOHR  # 0.52917721092


def build_uff_radii_table():
    """UFF 半径 (Bohr)，未含缩放因子"""
    table = np.zeros(118)
    for z, r in UFF_RADII_ANG.items():
        table[z] = r / BOHR
    return table


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
                # ================= 最小更改开始 =================
                # 获取原始值 (Hartree)
                raw_pred = float(outputs[key])
                raw_label = float(target[key])

                # 打印对比：显示 Ha 和转换后的 eV
                print(f"[{key} DEBUG]")
                print(f"  Pred : {raw_pred:.6f} Ha  =>  {raw_pred * Hartree:.6f} eV")
                print(f"  Label: {raw_label:.6f} Ha  =>  {raw_label * Hartree:.6f} eV")
                print(f"  Diff : {abs(raw_pred - raw_label):.6f} Ha  =>  {abs(raw_pred - raw_label) * Hartree:.6f} eV")
                # ================= 最小更改结束 =================
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

    # ================= 更改开始: 能量指标输出映射 =================
    # Orbital Energies (AI DM vs Gaussian Ref)
    if "HOMO" in data:
        processed["HOMO (eV)"] = data["HOMO"]
    if "LUMO" in data:
        processed["LUMO (eV)"] = data["LUMO"]
    if "GAP" in data:
        processed["GAP (eV)"] = data["GAP"]

    # Orbital Energies (Target DM via PySCF vs Gaussian Ref) -> 验证 PySCF 误差
    if "pyscf_HOMO" in data:
        processed["PySCF-HOMO-Err (eV)"] = data["pyscf_HOMO"]
    if "pyscf_LUMO" in data:
        processed["PySCF-LUMO-Err (eV)"] = data["pyscf_LUMO"]
    if "pyscf_GAP" in data:
        processed["PySCF-GAP-Err (eV)"] = data["pyscf_GAP"]

    if "ai_pyscf_HOMO" in data:
        processed["AI-PySCF-HOMO-Err (eV)"] = data["ai_pyscf_HOMO"]
    if "ai_pyscf_LUMO" in data:
        processed["AI-PySCF-LUMO-Err (eV)"] = data["ai_pyscf_LUMO"]
    if "ai_pyscf_GAP" in data:
        processed["AI-PySCF-GAP-Err (eV)"] = data["ai_pyscf_GAP"]

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


def get_electronic_properties(
        mol, ham=None, overlap=None, dm=None, shifted_ham=None, pcm_eps=1.0, mf=None
):
    """
    Helper function to extract electronic properties (Energies, Orbitals, Gap)
    from either Hamiltonian+Overlap OR Density Matrix.

    Updated to support PCM solvent model via pcm_eps.
    Now supports passing a pre-initialized PySCF mean-field object (`mf`).
    """
    from pyscf import dft  # 确保局部有引入
    # 1. 准备 Hamiltonian 和 Overlap
    if ham is None:
        if dm is None:
            raise ValueError("Must provide either Hamiltonian or Density Matrix")

        # 如果没有传入初始化的 mf，则在这里创建
        if mf is None:
            mf = dft.RKS(mol)
            mf.xc = "b3lyp"

            # --- 优化: 直接判断是否大于 1.0 即可 ---
            if pcm_eps > 1.0:
                mf = mf.PCM()
                mf.with_solvent.eps = pcm_eps
                mf.with_solvent.method = 'IEF-PCM'
                uff_radii = build_uff_radii_table()  # Bohr, 未缩放
                mf.with_solvent.radii_table = 1.1 * uff_radii  # Alpha=1.1, 已含缩放
                mf.with_solvent.lebedev_order = 31

        # PySCF get_fock returns (N, N) for RKS usually, but let's handle potential (1, N, N)
        # With PCM enabled, get_fock includes: H_core + J + K + V_pcm
        # 使用传入的或者新建的 mf 计算 Fock
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
    results = {
        "HOMO": energies[homo_idx],
        "LUMO": energies[homo_idx + 1],
        "GAP": energies[homo_idx + 1] - energies[homo_idx],
        "hamiltonian": ham_in_3d,  # (1, N, N)
        "overlap": ov_in_3d,  # (1, N, N)
        "shifted_ham": shifted_ham_2d,  # (N, N)
        "density_matrix": dm if dm is not None else make_rdm1(mo_coeff=coeffs, mo_occ=mo_occ),
        "mo_occ": mo_occ,
        "mo_energy": energies,
        "mo_coeff": coeffs,
        "orbital_coefficients": coeffs[:, : homo_idx + 1],
        "HOMO_coefficients": coeffs[:, homo_idx],
        "LUMO_coefficients": coeffs[:, homo_idx + 1],
        "occupied_orbital_energy": energies[: homo_idx + 1],
    }
    return results


def calculate_properties_from_dm(
        mol,
        dm,
        prefix,
        gen_dm_flag: bool = False,
        mf=None,
        fock=None,
        overlap=None,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        xc: str = "b3lyp",
        pcm_eps: float = 1.0,  # 优化：默认为真空介电常数 1.0
):
    """
    从 density matrix 生成 fchk，并用 Multiwfn 计算:
      - ESP max/min
      - Li 的 deformation factor (phi)

    优化:
    - 若上游(get_ham_flag=True)已算过 fock/overlap 或已有 mf，则可传入 mf/fock/overlap
      以跳过重复 mf 初始化与 mf.get_fock(dm=dm) 的耗时步骤。
    - 若还额外提供 mo_energy/mo_coeff，则连 eig 也可跳过。

    返回:
      (ESP_max_eV, ESP_min_eV, li_phi_or_None)
    """
    import json
    import numpy as np
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft
    from emoles.multiwfn import ESPCalculator, ELFDeformationCalculator

    # ========== 0) 准备 mf ==========
    if mf is None:
        mf = dft.RKS(mol)
        mf.xc = xc

        # 优化: 简化判断逻辑
        if pcm_eps > 1.0:
            mf = mf.PCM()
            mf.with_solvent.eps = pcm_eps
            mf.with_solvent.method = "IEF-PCM"
            uff_radii = build_uff_radii_table()
            mf.with_solvent.radii_table = 1.1 * uff_radii
            mf.with_solvent.lebedev_order = 31

    # ========== 1) overlap / fock / (mo_energy, mo_coeff) ==========
    if overlap is None:
        try:
            overlap = mf.get_ovlp()
        except Exception:
            overlap = mol.intor("int1e_ovlp")

    ov_2d = overlap[0] if (hasattr(overlap, "ndim") and overlap.ndim == 3) else overlap

    if (mo_energy is None) or (mo_coeff is None):
        if fock is None:
            fock = mf.get_fock(dm=dm)

        fock_2d = fock[0] if (hasattr(fock, "ndim") and fock.ndim == 3) else fock
        mo_energy, mo_coeff = mf.eig(fock_2d, ov_2d)

    if mo_occ is None:
        n_electrons = mol.tot_electrons()
        homo_idx = int(n_electrons / 2) - 1
        mo_occ = get_mo_occ(full_len=len(mo_energy), occ_len=homo_idx + 1)

    mf.mo_energy = np.array(mo_energy)
    mf.mo_coeff = np.array(mo_coeff)
    mf.mo_occ = np.array(mo_occ)
    mf.dm = dm

    # ========== 2) 生成 fchk ==========
    fch_filename = f"{prefix}.fch"
    fchk(mf, fch_filename, density=True)

    # ========== 3) ESP ==========
    esp_calculator = ESPCalculator(fch_filename)
    esp_results = esp_calculator.get_ESP_value()
    if gen_dm_flag:
        esp_calculator.get_acc_grid_data()

    with open(f"{prefix}_esp_info.json", "w") as f:
        json.dump(esp_results, fp=f, indent=4)

    esp_max = esp_results.get("ESP_max_eV", 0.0)
    esp_min = esp_results.get("ESP_min_eV", 0.0)

    # ========== 4) Li deformation factor (phi) ==========
    li_phi = None
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords_ang = mol.atom_coords(unit="Ang")
    li_indices_0based = [i for i, s in enumerate(symbols) if s == "Li"]

    if li_indices_0based:
        target_li_idx = li_indices_0based[0]
        i_1based = target_li_idx + 1
        li_center = coords_ang[target_li_idx]

        elf_calculator = ELFDeformationCalculator(
            fch_filename, isovalue=0.5, diff_list=[0.09], li_cutoff=1.1
        )
        save_id = f"{prefix}_{i_1based}"

        try:
            elf_res = elf_calculator.calculate(
                atom_index_1based=i_1based,
                li_center=li_center,
                li_id=save_id,
                radius=3.0,
                grid_spacing=0.1,
            )
            target_diff = 0.09
            if elf_res and target_diff in elf_res:
                li_phi = elf_res[target_diff]["phi"]
        except Exception as e:
            print(f"Warning: Failed to calculate phi for {prefix} Li: {e}")

    return esp_max, esp_min, li_phi


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
        n_save_cube_items: int = 5,
        temp_data_file: str = "temp_cube_data.pkl",
        max_items: int = 300,
        gen_esp_cube_flag: bool = False,
        summary_filename="evaluation_summary.npz",
        pcm_eps: float = 1,  # 外部传入的默认目标溶剂介电常数
        verbose_profiling: bool = False,
):
    import time
    import json
    import numpy as np
    import traceback
    import pickle
    import os
    from ase.db import connect
    from ase.io import write
    from tqdm import tqdm
    import pyscf
    from ase.units import Hartree

    def _log_time(msg):
        if verbose_profiling:
            print(msg)

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
    temp_cube_data = []

    def _load_npy_safe(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    # 优化点：在主循环外部只构建一次全局 UFF 半径表，极大减少内部开销
    if get_ham_flag:
        global_uff_radii_tb = build_uff_radii_table()

    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            if idx == max_items:
                break
            attempted_count += 1
            cwd_ = os.getcwd()

            t_item_start = time.time()
            _log_time(f"\n[{idx}] --- Timing Profiling Started ---")

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

                t_load = time.time()
                _log_time(f"[{idx}] [Time] NPY Load & Matrix Transform: {t_load - t_item_start:.4f} s")

                current_mol_charge = a_row.data.get("charge", mol_charge)
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

                t_mol = time.time()
                _log_time(f"[{idx}] [Time] PySCF Mole Build & Overlap: {t_mol - t_load:.4f} s")

                expected_electrons = float(total_electrons)
                Ne_pred = get_electron_number_from_dm(pred_dm, overlap)
                Ne_target = get_electron_number_from_dm(target_dm, overlap)

                # 1) 基础误差
                errors = calculate_dm_dipole_mae(pred_dm, target_dm, mol)

                _, atom_in_mo_indices = generate_molecule_transform_indices(
                    atom_types=an_atoms.get_chemical_symbols(),
                    atom_to_transform_indices=atom_to_transform_indices,
                )
                dm_diff = np.abs(pred_dm - target_dm)
                dm_diag, dm_non_diag = cut_and_cal_matrix(
                    full_matrix=dm_diff,
                    atom_in_mo_indices=atom_in_mo_indices
                )
                errors["diagonal_density_matrix_mae"] = dm_diag
                errors["non_diagonal_density_matrix_mae"] = dm_non_diag

                errors["pred_electron_number_error"] = abs(Ne_pred - expected_electrons)
                errors["target_electron_number_error"] = abs(Ne_target - expected_electrons)
                errors["electron_number_pred_vs_target_error"] = abs(Ne_pred - Ne_target)

                t_basic_err = time.time()
                _log_time(f"[{idx}] [Time] Basic DM & Dipole Metrics:  {t_basic_err - t_mol:.4f} s")

                electronic_properties_eV = None
                mf_gas = None
                pred_props_gas = None
                target_props_gas = None

                # 2) DM 推导的电子性质 / Ham / 轨道等
                if get_ham_flag:
                    # 安全读取 EPS，如果是气相数据或为0/负数，强制规范为 1.0
                    raw_eps = a_row.data.get("dielectric_constant", pcm_eps)
                    current_pcm_eps = float(raw_eps) if raw_eps is not None else 1.0
                    if current_pcm_eps < 1.0:
                        current_pcm_eps = 1.0

                    # --- [GAS] 提取本征形貌 ---
                    mf_gas = pyscf.dft.RKS(mol)
                    mf_gas.xc = "b3lyp"

                    pred_props_gas = get_electronic_properties(mol, dm=pred_dm, overlap=overlap, mf=mf_gas)
                    target_props_gas = get_electronic_properties(mol, dm=target_dm, overlap=overlap, mf=mf_gas)

                    t_mf = time.time()
                    _log_time(f"[{idx}] [Time] Init & Calc Gas Props: {t_mf - t_basic_err:.4f} s")

                    # --- [PCM] 提速核心逻辑优化 ---
                    if current_pcm_eps > 1.0:
                        mf_pcm = pyscf.dft.RKS(mol)
                        mf_pcm.xc = "b3lyp"
                        mf_pcm = mf_pcm.PCM()
                        mf_pcm.with_solvent.eps = current_pcm_eps
                        mf_pcm.with_solvent.method = 'IEF-PCM'
                        mf_pcm.with_solvent.radii_table = 1.1 * global_uff_radii_tb  # 直接使用外部全局表
                        mf_pcm.with_solvent.lebedev_order = 31

                        pred_props_pcm = get_electronic_properties(mol, dm=pred_dm, overlap=overlap, mf=mf_pcm)
                        target_props_pcm = get_electronic_properties(mol, dm=target_dm, overlap=overlap, mf=mf_pcm)
                    else:
                        # 若未启用溶剂，直接将 PCM 的引用指向 GAS 结果，彻底避免无意义的重复哈密顿量计算
                        pred_props_pcm = pred_props_gas
                        target_props_pcm = target_props_gas

                    t_prop = time.time()
                    _log_time(f"[{idx}] [Time] Check & Calc PCM Props:     {t_prop - t_mf:.4f} s")

                    # Gaussian label (eV)
                    gaussian_homo = a_row.data.get("HOMO_eV", 0.0)
                    gaussian_lumo = a_row.data.get("LUMO_eV", 0.0)
                    gaussian_gap = a_row.data.get("GAP_eV", gaussian_lumo - gaussian_homo)

                    pred_homo_ev = float(pred_props_pcm["HOMO"]) * Hartree
                    pred_lumo_ev = float(pred_props_pcm["LUMO"]) * Hartree
                    pred_gap_ev = float(pred_props_pcm["GAP"]) * Hartree

                    pyscf_homo_ev = float(target_props_pcm["HOMO"]) * Hartree
                    pyscf_lumo_ev = float(target_props_pcm["LUMO"]) * Hartree
                    pyscf_gap_ev = float(target_props_pcm["GAP"]) * Hartree

                    electronic_properties_eV = {
                        "pred": {
                            "HOMO_eV": pred_homo_ev,
                            "LUMO_eV": pred_lumo_ev,
                            "GAP_eV": pred_gap_ev,
                        },
                        "pyscf": {
                            "HOMO_eV": pyscf_homo_ev,
                            "LUMO_eV": pyscf_lumo_ev,
                            "GAP_eV": pyscf_gap_ev,
                        },
                        "gaussian": {
                            "HOMO_eV": float(gaussian_homo),
                            "LUMO_eV": float(gaussian_lumo),
                            "GAP_eV": float(gaussian_gap),
                        },
                    }

                    errors["HOMO"] = abs(pred_homo_ev - gaussian_homo)
                    errors["LUMO"] = abs(pred_lumo_ev - gaussian_lumo)
                    errors["GAP"] = abs(pred_gap_ev - gaussian_gap)

                    errors["pyscf_HOMO"] = abs(pyscf_homo_ev - gaussian_homo)
                    errors["pyscf_LUMO"] = abs(pyscf_lumo_ev - gaussian_lumo)
                    errors["pyscf_GAP"] = abs(pyscf_gap_ev - gaussian_gap)

                    errors["ai_pyscf_HOMO"] = abs(pred_homo_ev - pyscf_homo_ev)
                    errors["ai_pyscf_LUMO"] = abs(pred_lumo_ev - pyscf_lumo_ev)
                    errors["ai_pyscf_GAP"] = abs(pred_gap_ev - pyscf_gap_ev)

                    # Criterion 对比：必须使用 GAS，避免将 PCM 带来的非物理形变纳入损失评估
                    eval_keys = [
                        "hamiltonian",
                        "orbital_coefficients",
                        "HOMO_coefficients",
                        "LUMO_coefficients",
                    ]
                    ham_orb_errors = criterion(
                        pred_props_gas,
                        target_props_gas,
                        eval_keys,
                        flag=False,
                        atoms=an_atoms,
                        mol=mol,
                    )
                    errors.update(ham_orb_errors)

                    t_crit = time.time()
                    _log_time(f"[{idx}] [Time] Criterion Matrix Ops:       {t_crit - t_prop:.4f} s")

                    if idx < n_save_cube_items:
                        mol_info = {
                            "atom_nums": [int(x) for x in atom_nums],
                            "atom_coords": [list(at.position) for at in an_atoms],
                            "charge": int(current_mol_charge),
                            "spin": int(mol_spin),
                            "basis": basis,
                            "unit": "ang",
                        }
                        cube_item = {
                            "idx": idx,
                            "HOMO_sim": errors.get("HOMO_coefficients", 0.0),
                            "mol_info": mol_info,
                            "outputs": pred_props_gas,  # 保存本征轨道用于画图
                            "tgt_info": target_props_gas,
                        }
                        temp_cube_data.append(cube_item)
                else:
                    t_crit = time.time()
                    current_pcm_eps = pcm_eps

                # 3) ESP / deformation
                if get_esp_sta_flag:
                    if get_ham_flag and (pred_props_gas is not None) and (target_props_gas is not None):
                        p_esp_max, p_esp_min, p_phi = calculate_properties_from_dm(
                            mol,
                            pred_dm,
                            "pred",
                            gen_dm_flag=gen_esp_cube_flag,
                            mf=mf_gas,
                            fock=pred_props_gas.get("hamiltonian", None),
                            overlap=pred_props_gas.get("overlap", overlap),
                            mo_energy=pred_props_gas.get("mo_energy", None),
                            mo_coeff=pred_props_gas.get("mo_coeff", None),
                            mo_occ=pred_props_gas.get("mo_occ", None),
                        )
                        t_esp_max, t_esp_min, t_phi = calculate_properties_from_dm(
                            mol,
                            target_dm,
                            "target",
                            gen_dm_flag=gen_esp_cube_flag,
                            mf=mf_gas,
                            fock=target_props_gas.get("hamiltonian", None),
                            overlap=target_props_gas.get("overlap", overlap),
                            mo_energy=target_props_gas.get("mo_energy", None),
                            mo_coeff=target_props_gas.get("mo_coeff", None),
                            mo_occ=target_props_gas.get("mo_occ", None),
                        )
                    else:
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

                t_esp = time.time()
                if get_esp_sta_flag:
                    _log_time(f"[{idx}] [Time] ESP & Deformation Calc:     {t_esp - t_crit:.4f} s")

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
                local_result = {
                    "idx": idx,
                    "mol_info": mol_info_log,
                    "errors": errors,
                    "dielectric_constant_used": current_pcm_eps
                }
                if electronic_properties_eV is not None:
                    local_result["electronic_properties_eV"] = electronic_properties_eV

                with open("dm_evaluation_result.json", "w") as f_json:
                    json.dump(local_result, f_json, indent=4, default=str)

                flat_data = {"idx": idx}
                flat_data.update(errors)
                summary_data_list.append(flat_data)

                t_save = time.time()
                _log_time(f"[{idx}] [Time] Logging & File Saving:      {t_save - t_esp:.4f} s")
                _log_time(f"[{idx}] [Time] >>> TOTAL For Item {idx} <<< : {t_save - t_item_start:.4f} s\n")

            except Exception as e:
                fail_count += 1
                failed_indices.append(idx)
                traceback.print_exc()
                print(f"[evaluate_dm_from_npy] idx {idx} failed: {repr(e)}")
            finally:
                os.chdir(cwd_)

    if temp_data_file and len(temp_cube_data) > 0:
        save_path = os.path.join(npy_folder_path, temp_data_file)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(temp_cube_data, f)
        except Exception as e:
            print(f"[evaluate_dm_from_npy] Failed to save temp cube data: {e}")

    n = total_error_dict["total_items"]
    if n > 0:
        for key in list(total_error_dict["pred_vs_label"].keys()):
            total_error_dict["pred_vs_label"][key] /= n

    end_time = time.time()
    total_error_dict["second_per_item"] = (end_time - start_time) / max(1, n)

    if len(summary_data_list) > 0:
        all_keys = set().union(*(d.keys() for d in summary_data_list))
        npz_dict = {}
        for key in all_keys:
            values = []
            for item in summary_data_list:
                values.append(item.get(key, np.nan))
            npz_dict[key] = np.array(values)
        np.savez(os.path.join(npy_folder_path, summary_filename), **npz_dict)

    final_data = {"pred_vs_label": total_error_dict["pred_vs_label"].copy()}
    final_data["pred_vs_label"]["second_per_item"] = total_error_dict["second_per_item"]

    result_dict = process_dm_loss_dict(final_data, key="pred_vs_label")
    result_dict.update({
        "Total Items": n,
        "Attempted Items": attempted_count,
        "Failed Items": fail_count
    })
    return result_dict


def get_mae_from_npy(
        abs_ase_path,
        npy_folder_path,
        temp_data_file=None,
        united_overlap_flag=False,
        convention="def2svp",
        mol_charge=0,
        save_summary=False,
        full_save_items=10,
        pcm_eps: float = 25.59,  # Default solvent epsilon
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

            shifted_label_ham = None

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

            # --- 使用 Helper 函数获取属性 (传入 pcm_eps) ---
            # 注: 如果传入了明确的 ham, get_electronic_properties 内部其实不会重新构建 Fock，
            # 而是直接使用传入的 ham。这里传入 pcm_eps 主要是为了 API 统一性以及如果内部有
            # 基于 mf 的操作时能保持一致。
            outputs = get_electronic_properties(
                mol, ham=pred_ham_prep, overlap=pred_ov_prep, pcm_eps=pcm_eps
            )
            tgt_info = get_electronic_properties(
                mol,
                ham=orig_ham_prep,
                overlap=orig_ov_prep,
                shifted_ham=shifted_label_ham,
                pcm_eps=pcm_eps
            )

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
            total_error_dict["pred_vs_label"][key] /= n

    end_time = time.time()
    total_error_dict["second_per_item"] = (end_time - start_time) / max(1, n)

    total_error_dict = process_loss_dict(total_error_dict, key="pred_vs_label")
    print(total_error_dict)

    if save_summary and temp_data_file is not None:
        with open(temp_data_file, "wb") as f:
            pickle.dump(temp_data, f)

    return total_error_dict