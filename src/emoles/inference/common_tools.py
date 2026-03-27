import os

import json
import numpy as np

from ase.units import Hartree
from emoles.utils import matrix_transform
from ase.db.core import connect
from ase.io import write
from ase.atom import Atom
from ase.atoms import Atoms
import pandas as pd
import ase

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from emoles.multiwfn import ESPCalculator
from rdkit.Chem.rdDetermineBonds import DetermineBonds
import torch
from collections import defaultdict


def extract_model_params(model):
    """
    从 dptb 模型中提取 basis 和 r_max 参数。
    包含 Debug 信息打印，并兼容 r_max 为统一标量的情况。
    """

    print("\n" + "=" * 20 + " DEBUG: Model Params Inspection " + "=" * 20)

    # --- Debug: 检查 Embedding 层 ---
    if hasattr(model, 'embedding'):
        embed = model.embedding
        print(f"[INFO] model.embedding type: {type(embed)}")
        print(f"[INFO] model.embedding keys/vars: {list(embed.__dict__.keys())}")

        if hasattr(embed, 'basis'):
            print(f"[INFO] found 'basis' in embedding: {embed.basis}")
        else:
            print("[WARN] 'basis' NOT found in embedding attributes!")
    else:
        print("[ERROR] Model has no attribute 'embedding'")
        return {}, {}

    # --- Debug: 检查 InitLayer 层 ---
    if hasattr(model.embedding, 'init_layer'):
        init_layer = model.embedding.init_layer
        print(f"[INFO] init_layer type: {type(init_layer)}")
        print(f"[INFO] init_layer keys/vars: {list(init_layer.__dict__.keys())}")

        # 重点检查 r_max 相关的属性
        r_max_scalar = getattr(init_layer, 'r_max', None)
        r_max_dict = getattr(init_layer, 'r_max_dict', None)

        print(f"[INFO] init_layer.r_max (scalar/tensor): {r_max_scalar}")
        print(f"[INFO] init_layer.r_max_dict (dict/None): {r_max_dict}")
    else:
        print("[ERROR] model.embedding has no attribute 'init_layer'")
        return {}, {}

    print("=" * 60 + "\n")

    # ================= 提取逻辑开始 =================

    # 1. 处理 basis (List of strings -> Dense string "3s2p1d")
    raw_basis = getattr(model.embedding, 'basis', {})
    basis_clean = {}
    orbital_types = ['s', 'p', 'd', 'f']

    for elem, orb_list in raw_basis.items():
        counts = defaultdict(int)
        for orb in orb_list:
            # 假设轨道格式为 "1s", "2p" 等，取最后一个字符
            if orb:
                o_type = orb[-1]
                counts[o_type] += 1

        dense_str = ""
        for o_type in orbital_types:
            count = counts[o_type]
            if count > 0:
                dense_str += f"{count}{o_type}"
        basis_clean[elem] = dense_str

    # 2. 处理 r_max (兼容 scalar 和 dict)
    # 重新获取引用，以防上面 debug 代码块作用域问题
    init_layer = model.embedding.init_layer
    raw_r_max_dict = getattr(init_layer, 'r_max_dict', None)
    raw_r_max_scalar = getattr(init_layer, 'r_max', None)

    r_max_clean = {}

    if raw_r_max_dict is not None:
        # 情况 A: r_max_dict 存在 (说明不同元素有不同 r_max)
        for elem, tensor_val in raw_r_max_dict.items():
            r_max_clean[elem] = tensor_val.item()

    elif raw_r_max_scalar is not None:
        # 情况 B: r_max_dict 为 None (说明使用了统一的 r_max)
        # 此时我们需要根据 basis 中的元素列表，赋予每个元素相同的 r_max
        scalar_val = raw_r_max_scalar.item()
        for elem in basis_clean.keys():
            r_max_clean[elem] = scalar_val

    else:
        print("[ERROR] Both r_max and r_max_dict are missing in InitLayer!")

    return basis_clean, r_max_clean


class info_collector:
    def __init__(self):
        self.homo_list = []
        self.lumo_list = []
        self.gap_list = []
    def parse_orbital_energies(self, orbital_energies, homo_index):
        homo = orbital_energies[homo_index]*Hartree
        lumo = orbital_energies[homo_index+1]*Hartree
        gap = lumo - homo
        self.homo_list.append(homo)
        self.lumo_list.append(lumo)
        self.gap_list.append(gap)
    def dump_to_csv(self, csv_path):
        data = {
            'HOMO': self.homo_list,
            'LUMO': self.lumo_list,
            'Gap': self.gap_list
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def prepare_np(overlap_matrix, full_hamiltonian, atom_numbers, transform_ham_flag=False, transform_overlap_flag=False, transform_convention='back2pyscf'):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(full_hamiltonian, atom_numbers, convention=transform_convention)
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(overlap_matrix, atom_numbers, convention=transform_convention)
    return full_hamiltonian, overlap_matrix


def get_overlap_matrix(ase_atoms, basis):
    import pyscf

    mol = pyscf.gto.Mole()
    t = [[ase_atoms.numbers[atom_idx], an_atom.position]
         for atom_idx, an_atom in enumerate(ase_atoms)]
    mol.build(verbose=0, atom=t, basis=basis, unit='ang')
    overlap = mol.intor("int1e_ovlp")
    return overlap, mol


def generate_cube_files(ase_db_path: str, out_path: str, n_grid, basis='def2svp', dm_flag=False, keep_xyz_file: bool=True, dm_grid: int=40, limit: int=2):
    from pyscf.scf.hf import make_rdm1, dip_moment
    from pyscf import tools

    """Generate cube files for HOMO and LUMO orbitals and save them in sub-folders named by idx."""
    if basis == 'def2svp':
        transform_convention = 'back2pyscf'
        overlap_basis = 'def2svp'
    elif basis == '6311gdp':
        transform_convention = 'back_2_thu_pyscf'
        overlap_basis = '6-311+g(d,p)'
    else:
        raise NotImplementedError
    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, 'cube')
    with connect(ase_db_path) as db:
        for idx, a_row in enumerate(db.select()):
            an_atoms = a_row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load('predicted_ham.npy')
            hamiltonian, overlap = prepare_np(overlap_matrix=overlap, full_hamiltonian=predicted_ham, atom_numbers=an_atoms.numbers, transform_ham_flag=True, transform_overlap_flag=False, transform_convention=transform_convention)
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(overlap_matrix=overlap, full_hamiltonian=hamiltonian)
            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(orbital_energies=orbital_energies, homo_index=homo_idx)
            HOMO_coefficients, LUMO_coefficients = orbital_coefficients[:, homo_idx], orbital_coefficients[:, homo_idx+1]
            # tools.cubegen.orbital(mol, 'HOMO_big_margin.cube', HOMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid, margin=9)
            # tools.cubegen.orbital(mol, 'LUMO_big_margin.cube', LUMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid, margin=9)
            tools.cubegen.orbital(mol, 'HOMO.cube', HOMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'LUMO.cube', LUMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)

            if dm_flag:
                mo_occ = np.zeros(overlap.shape[-1])
                mo_occ[:homo_idx+1] = 2
                dm = make_rdm1(mo_coeff=orbital_coefficients, mo_occ=mo_occ)
                tools.cubegen.density(mol, 'electron_density.cube', dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                tools.cubegen.mep(mol, 'molecular_electrostatic_potential.cube', dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                mol_dip = dip_moment(mol, dm, unit='DEBYE')
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                dipole_info = {
                    'Dipole_Moment_Vector_DEBYE': mol_dip.tolist(),
                    'Dipole_Moment_Norm_DEBYE': float(dip_magnitude),
                }
                with open('dipole_info.json', 'w') as f:
                    json.dump(dipole_info, fp=f)

            if keep_xyz_file:
                write('atomic_structure.xyz', an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, 'energy_info.csv')
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)


def calculate_with_multiwfn(ase_db_path: str, out_path: str, n_grid, basis='def2svp', esp_flag=False, keep_xyz_file: bool=True, dm_grid: int=40, limit: int=2):
    from pyscf.scf.hf import make_rdm1, dip_moment
    from pyscf import tools, dft
    from emoles.multiwfn import ESPCalculator
    from mokit.lib.py2fch_direct import fchk

    """Generate cube files for HOMO and LUMO orbitals and save them in sub-folders named by idx."""
    if basis == 'def2svp':
        transform_convention = 'back2pyscf'
        overlap_basis = 'def2svp'
    elif basis == '6311gdp':
        transform_convention = 'back_2_thu_pyscf'
        overlap_basis = '6-311+g(d,p)'
    else:
        raise NotImplementedError
    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, 'cube')
    with connect(ase_db_path) as db:
        for idx, a_row in enumerate(db.select()):
            an_atoms = a_row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load('predicted_ham.npy')
            hamiltonian, overlap = prepare_np(overlap_matrix=overlap, full_hamiltonian=predicted_ham, atom_numbers=an_atoms.numbers, transform_ham_flag=True, transform_overlap_flag=False, transform_convention=transform_convention)
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(overlap_matrix=overlap, full_hamiltonian=hamiltonian)
            mf = dft.RKS(mol)
            mf.mo_coeff = orbital_coefficients
            mf.mo_energy = orbital_energies
            fchk(mf, 'predicted.fch', density=True)

            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(orbital_energies=orbital_energies, homo_index=homo_idx)
            # HOMO_coefficients, LUMO_coefficients = orbital_coefficients[:, homo_idx], orbital_coefficients[:, homo_idx+1]

            if esp_flag:
                esp_calculator = ESPCalculator("predicted.fch")
                esp_results, cube_file = esp_calculator.calculate_grid_data()
                with open('esp_info.json', 'w') as f:
                    json.dump(esp_results, fp=f)

            if keep_xyz_file:
                write('atomic_structure.xyz', an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, 'energy_info.csv')
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)



def calculate_esp_from_dm(mol, dm, prefix, gen_dm_flag=False):
    """
    A helper function to
    1. generate an fchk file from a density matrix with help of mokit
    2. compute ESP properties using Multiwfn.

    Parameters:
    -----------
    mol : gto.Mole
        PySCF molecule object.
    dm : np.ndarray
        Density matrix.
    prefix : str
        A prefix for all output files (e.g., "predicted", "target").

    Returns:
    --------
    tuple
        A tuple containing (ESP_max_in_eV, ESP_min_in_eV).
    """
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft, tools

    # 1. Set up and run PySCF DFT calculation from the density matrix
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    fock = mf.get_fock(dm=dm)  # Build Fock matrix using the given density matrix
    s = mf.get_ovlp()  # Get the overlap matrix

    # Solve the generalized eigenvalue problem to get orbital energies and coefficients
    orbital_energies, orbital_coefficients = mf.eig(fock, s)

    # Store results in the mean-field object for fchk export
    mf.mo_energy = orbital_energies
    mf.mo_coeff = orbital_coefficients
    mf.dm = dm

    # 2. Generate a formatted checkpoint (.fchk) file
    fch_filename = f"{prefix}.fch"
    fchk(mf, fch_filename, density=True)

    # 3. Initialize ESPCalculator and run Multiwfn tasks
    esp_calculator = ESPCalculator(fch_filename)

    # 3a. Get ESP surface extrema (values are in eV)
    esp_results = esp_calculator.get_ESP_value()

    # 3b. Generate high-quality grid files (density.cub and totesp_ev.cub)
    # Note: The function handles renaming the files automatically
    if gen_dm_flag:
        grid_files = esp_calculator.get_acc_grid_data()

    # 4. Save the parsed ESP extrema info to a JSON file for record-keeping
    json_filename = f"{prefix}_esp_info.json"
    with open(json_filename, 'w') as f:
        json.dump(esp_results, fp=f, indent=4)

    # 5. Extract and return the max and min ESP values
    esp_max = esp_results.get('ESP_max_eV', 0)  # Use .get for safe access
    esp_min = esp_results.get('ESP_min_eV', 0)

    return esp_max, esp_min


def mol_2_atom(mol: rdkit.Chem.rdchem.Mol):
    conf = mol.GetConformer()
    an_atoms = Atoms()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        atom = mol.GetAtoms()[i]
        a_symbol = atom.GetSymbol()
        an_new_atom = Atom(symbol=a_symbol, position=(position.x, position.y, position.z))
        an_atoms.append(an_new_atom)
    return an_atoms


def atom_2_mol(an_atoms: ase.atoms.Atoms):
    write(filename='temp.xyz', images=an_atoms)
    raw_mol = Chem.MolFromXYZFile('temp.xyz')
    mol = Chem.Mol(raw_mol)
    DetermineBonds(mol, useHueckel=True)
    os.remove('temp.xyz')
    return mol


def smile_to_inchi(smile: str) -> str:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smile}")
    return Chem.MolToInchi(mol)


def smile_to_maccs_fp_arr(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(list(fingerprint.ToBitString())).astype(int)


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    a, b = fp1.astype(bool), fp2.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def atom_2_smile(an_atoms: ase.atoms.Atoms):
    a_mol = atom_2_mol(an_atoms)
    a_mol = Chem.RemoveHs(a_mol)
    a_smile = Chem.MolToSmiles(a_mol, isomericSmiles=False)
    return a_smile


def smile_2_atom(smile: str, maxAttempts: int=1000000):
    a_mol = Chem.MolFromSmiles(smile)
    a_mol_with_H = Chem.AddHs(a_mol)
    AllChem.EmbedMolecule(a_mol_with_H, useRandomCoords=True, maxAttempts=maxAttempts)
    AllChem.MMFFOptimizeMolecule(a_mol_with_H)
    an_atoms = mol_2_atom(mol=a_mol_with_H)
    return an_atoms


def annotate_db_dc_by_similarity(q_db_path: str, ref_db_path: str):
    """
    独立的核心逻辑（针对单分子 DB）：
    1. 预加载 ref_db 的 InChI 和 MACCS 指纹
    2. 遍历 q_db，生成 SMILES 和 InChI
    3. 优先精确匹配 InChI，否则降级使用指纹相似度匹配获取 DC
    4. 更新 q_db
    """

    # 1. 预加载 Reference DB
    ref_rows = []
    inchi_map = {}
    with connect(ref_db_path) as ref_db:
        for row in ref_db.select():
            inchi = getattr(row, "inchi", None)
            dc = float(getattr(row, "dielectric_constant", 0.0))
            fp = np.array(row.data["maccs_fp"]).astype(int)

            ref_dict = {"dc": dc, "fp": fp}
            ref_rows.append(ref_dict)
            if inchi:
                inchi_map[inchi] = ref_dict

    # 2. 遍历并更新 Query DB
    with connect(q_db_path) as q_db:
        for row in tqdm(q_db.select(), desc="Annotating weighted DC"):
            data = dict(row.data) if row.data else {}

            # 获取分子的 SMILES 和 InChI（统一使用复数形式的 smiles 变量名）
            an_atoms = row.toatoms()
            smiles = atom_2_smile(an_atoms)
            inchi = smile_to_inchi(smiles)

            # 3. 匹配介电常数 (DC)
            if inchi in inchi_map:
                # 优先 InChI 精确匹配
                matched_dc = inchi_map[inchi]["dc"]
            else:
                # 降级：指纹 Tanimoto 相似度最大值匹配
                q_fp = smile_to_maccs_fp_arr(smiles)
                best_match = max(ref_rows, key=lambda r: tanimoto_similarity(q_fp, r["fp"]))
                matched_dc = best_match["dc"]

            # 4. 更新 Query DB 数据库
            data["dielectric_constant_weighted"] = matched_dc
            q_db.update(row.id, dielectric_constant=matched_dc, data=data)


def smile_2_db(smile_path: str, db_path: str, fail_smile_path: str,  maxAttempts: int=1000000):
    print('Each row corresponds to a smile by default.')
    real_count = 0
    fail_count = 0
    with connect(db_path) as db, open(smile_path, 'r') as smi_r:
        for i, a_line in enumerate(smi_r.readlines()):
            a_smile = a_line.strip()
            if a_smile == '':
                print('An empty line is detected.')
                continue
            try:
                an_atoms = smile_2_atom(smile=a_smile, maxAttempts=maxAttempts)
                db.write(atoms=an_atoms, smile=a_smile)
                real_count = real_count + 1
            except Exception as e:
                print(e)
                fail_count = fail_count + 1
                with open(fail_smile_path, 'a') as f_f:
                    f_f.write(a_smile)
                    f_f.write('\n')
    print(f'real: {real_count}')
    print(f'fail: {fail_count}')

