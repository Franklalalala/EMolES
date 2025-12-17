import os
import shutil
import json
import time
import pickle
import traceback

import numpy as np
import pandas as pd
import lmdb
import torch
from ase.db import connect
from tqdm import tqdm
from ase.units import Hartree

# PySCF imports
import pyscf
from pyscf import gto, dft, tools
from pyscf.scf.hf import dip_moment, make_rdm1

# DPTB imports (Based on your provided file 2)
from dftio.data import _keys
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import collect_cutoffs

# EMOLES imports (Based on your provided file 1)
from emoles.utils import (
    matrix_transform,
    get_mo_occ,
)
from emoles.inference.common_tools import atom_2_smile, calculate_esp_from_dm, extract_model_params


# ==========================================
# Helper Functions from Loss File
# ==========================================

def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    """
    Solve generalized eigenvalue problem HC = SCE.
    Input matrices should be 3D (Batch, N, N) or 2D (N, N).
    Returns energies and coefficients.
    """
    # Ensure inputs are 3D for consistent processing if they come in as 2D
    if overlap_matrix.ndim == 2:
        overlap_matrix = overlap_matrix[None, ...]
    if full_hamiltonian.ndim == 2:
        full_hamiltonian = full_hamiltonian[None, ...]

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

    # Return the first item (assuming batch size 1 for inference scripts)
    return orbital_energies[0], orbital_coefficients[0]


def get_electronic_properties(mol, ham=None, overlap=None, dm=None):
    """
    Extract electronic properties (Energies, Orbitals, Gap) from Ham+Overlap.
    """
    # 1. Prepare Hamiltonian and Overlap
    if ham is None:
        if dm is None:
            raise ValueError("Must provide either Hamiltonian or Density Matrix")
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        ham = mf.get_fock(dm=dm)
        if overlap is None:
            overlap = mf.get_ovlp()

    if overlap is None:
        overlap = mol.intor("int1e_ovlp")

    # 2. Normalize dimensions to 2D for single molecule processing
    ham_2d = ham[0] if ham.ndim == 3 else ham
    ov_2d = overlap[0] if overlap.ndim == 3 else overlap

    # 3. Solve Generalized Eigenvalue Problem
    energies, coeffs = cal_orbital_and_energies(
        overlap_matrix=ov_2d, full_hamiltonian=ham_2d
    )

    # 4. Determine Occupation and Indices
    n_electrons = mol.tot_electrons()
    # For closed shell RKS
    homo_idx = int(n_electrons / 2) - 1
    lumo_idx = homo_idx + 1

    mo_occ = get_mo_occ(full_len=len(energies), occ_len=homo_idx + 1)

    results = {
        "HOMO": energies[homo_idx],
        "LUMO": energies[lumo_idx],
        "GAP": energies[lumo_idx] - energies[homo_idx],
        "hamiltonian": ham_2d,
        "overlap": ov_2d,
        "mo_occ": mo_occ,
        "orbital_coefficients": coeffs,  # Return full coeffs
        "HOMO_coefficients": coeffs[:, homo_idx],
        "LUMO_coefficients": coeffs[:, lumo_idx],
        "occupied_orbital_energy": energies[: homo_idx + 1],
        "all_energies": energies,
        "homo_idx": homo_idx,
        "lumo_idx": lumo_idx
    }
    return results


def _load_npy_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


# ==========================================
# DPTB Inference Logic
# ==========================================

def ase_db_2_dummy_dptb_lmdb(ase_db_path: str, dptb_lmdb_path: str):
    dptb_lmdb_path = os.path.join(dptb_lmdb_path, "data.{}.lmdb".format(os.getpid()))
    os.makedirs(dptb_lmdb_path)
    lmdb_env = lmdb.open(dptb_lmdb_path, map_size=1048576000000, lock=True)
    with connect(ase_db_path) as src_db:
        for idx, a_row in enumerate(src_db.select()):
            an_atoms = a_row.toatoms()
            data_dict = {
                _keys.ATOMIC_NUMBERS_KEY: an_atoms.numbers,
                _keys.PBC_KEY: np.array([False, False, False]),
                _keys.POSITIONS_KEY: an_atoms.positions.reshape(1, -1, 3).astype(np.float32),
                _keys.CELL_KEY: an_atoms.cell.reshape(1, 3, 3).astype(np.float32),
                "charge": a_row.data.get('charge', 0),
                "idx": idx,
                "nf": 0
            }
            data_dict = pickle.dumps(data_dict)
            entries = lmdb_env.stat()["entries"]
            with lmdb_env.begin(write=True) as txn:
                txn.put(entries.to_bytes(length=4, byteorder='big'), data_dict)
    lmdb_env.close()


def save_info_2_npy(folder_path, idx, batch_info, model, device, has_overlap):
    cwd_ = os.getcwd()
    os.chdir(folder_path)
    if not os.path.exists(str(idx)):
        os.makedirs(str(idx))
    os.chdir(str(idx))

    batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Save Hamiltonian
    a_ham_hr2hk = HR2HK_Gamma_Only(
        idp=model.idp,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device=device
    )
    ham_out_data = a_ham_hr2hk.forward(batch_info)
    a_ham = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
    ham_ndarray = a_ham.real.cpu().numpy()
    np.save('predicted.npy', ham_ndarray)  # Save as 2D if batch is 1 usually

    # Save Overlap if needed
    if has_overlap:
        an_overlap_hr2hk = HR2HK(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            overlap=True,
            device=device
        )
        overlap_out_data = an_overlap_hr2hk.forward(batch_info)
        an_overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
        overlap_ndarray = an_overlap.real.cpu().numpy()
        np.save('predicted_overlap.npy', overlap_ndarray)

    os.chdir(cwd_)


default_fine_tune_ckpt_path = r'/share/dptb_ckpt/fine_tune/best.pth'
default_fine_tune_input_json_path = r'/share/dptb_ckpt/fine_tune/input.json'


def dptb_infer_from_ase_db(ase_db_path: str, out_path: str,
                           checkpoint_path: str = default_fine_tune_ckpt_path,
                           limit: int = 200, device: str = 'cuda'):
    import e3nn
    e3nn.set_optimization_defaults(jit_script_fx=False)

    device = torch.device(device)
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    basis, r_max = extract_model_params(model)
    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)

    if os.path.exists(abs_out_path):
        shutil.rmtree(abs_out_path)
    os.makedirs(abs_out_path)

    lmdb_path = os.path.join(abs_out_path, 'lmdb')
    npy_path = os.path.join(abs_out_path, 'npy')
    os.makedirs(npy_path)

    ase_db_2_dummy_dptb_lmdb(ase_db_path, lmdb_path)

    reference_info = {
        "root": lmdb_path,
        "prefix": "data",
        "type": "LMDBDataset",
        "get_DM": False,
        "get_Hamiltonian": False,
        "get_overlap": False
    }
    reference_datasets = build_dataset(basis=basis, r_max=r_max, train_w_charge=True, **reference_info)
    reference_loader = DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)

    start_time = time.time()
    idx = 0
    for idx, a_ref_batch in enumerate(reference_loader):
        if idx >= limit:
            break
        batch = a_ref_batch.to(device)
        batch = AtomicData.to_AtomicDataDict(batch)
        with torch.no_grad():
            predicted_data = model(batch)
        # Note: Saving as predicted_ham.npy now to be explicit
        save_info_2_npy(folder_path=npy_path, idx=idx, batch_info=predicted_data, model=model, device=device, has_overlap=False)

    end_time = time.time()
    print('DPTB inference done.')
    second_per_item = (end_time - start_time) / max(1, min(idx + 1, limit))
    print(f'DPTB Inference Time (s/item): {second_per_item}')


# ==========================================
# Post-Processing: Density Matrix
# ==========================================

def get_dm_info_from_npy(ase_db_path,
                         npy_folder_path,
                         convert_smiles_flag=False,
                         convention='def2svp',
                         mol_charge=0,
                         pred_dm_filename='predicted.npy',  # Updated filename convention if needed
                         transform_dm_flag=True,
                         get_esp_sta_flag=True,
                         get_dm_cube_flag=False,
                         dm_cube_src='pyscf',
                         keep_xyz_file=True,
                         max_cube_save: int = 5,
                         max_items: int = 300,
                         dm_grid: int = 40,
                         ):
    print('Start DM postprocess')
    if convention == '6311gdp':
        basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
    else:
        basis = 'def2svp'
        back_convention = 'back2pyscf'

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()
    all_dm_info = []
    start_time = time.time()

    with connect(ase_db_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            if idx == max_items:
                break

            work_dir = os.path.join(npy_folder_path, f'{idx}')
            if not os.path.exists(work_dir):
                continue
            os.chdir(work_dir)

            try:
                atom_nums = a_row.numbers
                an_atoms = a_row.toatoms()
                if convert_smiles_flag:
                    smiles = atom_2_smile(an_atoms)

                # --- Charge & Spin Logic from Loss File ---
                current_mol_charge = a_row.data.get("charge", mol_charge)
                sum_of_atomic_numbers = an_atoms.get_atomic_numbers().sum()
                total_electrons = sum_of_atomic_numbers - current_mol_charge
                mol_spin = total_electrons % 2

                # Load and Transform DM
                pred_dm = _load_npy_safe(pred_dm_filename)
                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)

                # Build PySCF Mole
                mol = pyscf.gto.Mole()
                t = [[atom_nums[atom_idx], an_atom.position]
                     for atom_idx, an_atom in enumerate(an_atoms)]
                mol.charge = current_mol_charge
                mol.spin = mol_spin
                mol.build(verbose=0, atom=t, basis=basis, unit='ang')

                # Generate Cube Files
                multiwfn_gen_dm_flag = False
                if get_dm_cube_flag and idx < max_cube_save:
                    if dm_cube_src == 'pyscf':
                        tools.cubegen.density(mol, 'pred_electron_density.cube', pred_dm, nx=dm_grid, ny=dm_grid,
                                              nz=dm_grid)
                        tools.cubegen.mep(mol, 'pred_molecular_electrostatic_potential.cube', pred_dm, nx=dm_grid,
                                          ny=dm_grid, nz=dm_grid)
                    else:
                        multiwfn_gen_dm_flag = True

                # Calculate Properties
                pred_esp_max, pred_esp_min = 0, 0
                if get_esp_sta_flag:
                    # Logic assumes calculate_esp_from_dm is imported or defined
                    pred_esp_max, pred_esp_min = calculate_esp_from_dm(mol, pred_dm, "pred", multiwfn_gen_dm_flag)

                mol_dip = dip_moment(mol, pred_dm, unit='DEBYE')
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                to_sig4 = lambda x: float(f"{x:.4g}")

                dm_info = {
                    'Index': idx,
                    'SMILES': smiles,
                    'Charge': current_mol_charge,
                    'Dipole-X-Debye': to_sig4(mol_dip[0]),
                    'Dipole-Y-Debye': to_sig4(mol_dip[1]),
                    'Dipole-Z-Debye': to_sig4(mol_dip[2]),
                    'Dipole-Moment-magnitude-Debye': to_sig4(dip_magnitude),
                    'ESP-Max-eV': to_sig4(pred_esp_max),
                    'ESP-Min-eV': to_sig4(pred_esp_min),
                }
                if convert_smiles_flag:
                    dm_info.update({'SMILES': smiles})

                all_dm_info.append(dm_info)

                if keep_xyz_file:
                    from ase.io import write
                    write("atomic_structure.xyz", an_atoms)

                with open('dm_info.json', 'w') as f:
                    json.dump(dm_info, fp=f, indent=4)

            except Exception as e:
                print(f"Failed at idx {idx}: {e}")
                traceback.print_exc()

    end_time = time.time()
    os.chdir(cwd_)

    if all_dm_info:
        df = pd.DataFrame(all_dm_info)
        output_csv_path = os.path.join(cwd_, 'dm_summary.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved DM summary to {output_csv_path}")

    second_per_item = (end_time - start_time) / max(1, len(all_dm_info))
    print(f'DM Post-process Time (s/item): {second_per_item}')


# ==========================================
# Post-Processing: Hamiltonian -> Energy/Cube
# ==========================================

def get_ham_info_from_npy(ase_db_path,
                          npy_folder_path,
                          convert_smiles_flag=False,
                          convention='def2svp',
                          mol_charge=0,
                          pred_ham_filename='predicted.npy',
                          max_items: int = 300,
                          max_cube_save: int = 5,
                          cube_grid: int = 40):
    """
    Read Hamiltonian from npy -> Rotate -> Solve with PySCF Overlap ->
    Get Energies/Coeffs -> Draw HOMO/LUMO -> Save Summary.
    """
    print('Start Hamiltonian postprocess')

    if convention == '6311gdp':
        basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
    else:
        basis = 'def2svp'
        back_convention = 'back2pyscf'

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()
    all_ham_info = []
    start_time = time.time()

    with connect(ase_db_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            if idx >= max_items:
                break

            work_dir = os.path.join(npy_folder_path, f'{idx}')
            if not os.path.exists(work_dir):
                continue
            os.chdir(work_dir)

            try:
                atom_nums = a_row.numbers
                an_atoms = a_row.toatoms()
                if convert_smiles_flag:
                    smiles = atom_2_smile(an_atoms)

                # --- Charge & Spin Logic ---
                current_mol_charge = a_row.data.get("charge", mol_charge)
                sum_of_atomic_numbers = an_atoms.get_atomic_numbers().sum()
                total_electrons = sum_of_atomic_numbers - current_mol_charge
                mol_spin = total_electrons % 2

                # 1. Load Hamiltonian
                pred_ham = _load_npy_safe(pred_ham_filename)

                # 2. Transform Basis (Rotation)
                # Convention: Transform predicted matrix to PySCF basis order
                pred_ham_pyscf = matrix_transform(pred_ham, atom_nums, convention=back_convention)

                # 3. Build PySCF Mole & Get Overlap
                mol = pyscf.gto.Mole()
                t = [[atom_nums[atom_idx], an_atom.position]
                     for atom_idx, an_atom in enumerate(an_atoms)]
                mol.charge = current_mol_charge
                mol.spin = mol_spin
                mol.build(verbose=0, atom=t, basis=basis, unit='ang')

                overlap = mol.intor("int1e_ovlp")

                # 4. Solve for Energies and Coefficients
                # Uses the helper imported/defined from file 1
                props = get_electronic_properties(
                    mol, ham=pred_ham_pyscf, overlap=overlap, dm=None
                )

                # Convert to eV if needed, usually properties are in Hartree, let's keep consistent with Loss file which converts later or explicitly. Loss file says "HOMO (eV)": data["HOMO"]. Let's store raw Hartree here or convert.
                # Usually pyscf returns Hartree. The loss file output format converts to eV.
                # Here we save Hartree to be safe, or eV if preferred. Let's save eV for ease of reading.
                homo_ev = props["HOMO"] * 27.2114
                lumo_ev = props["LUMO"] * 27.2114
                gap_ev = props["GAP"] * 27.2114

                # 5. Draw HOMO / LUMO Cubes
                if idx < max_cube_save:
                    # PySCF cubegen needs the specific orbital coefficient vector (N,)
                    homo_coeff = props["HOMO_coefficients"]
                    lumo_coeff = props["LUMO_coefficients"]

                    tools.cubegen.orbital(mol, 'homo.cube', homo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)
                    tools.cubegen.orbital(mol, 'lumo.cube', lumo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)

                to_sig4 = lambda x: float(f"{x:.4g}")

                ham_info = {
                    'Index': idx,
                    'Charge': current_mol_charge,
                    'HOMO_eV': to_sig4(homo_ev),
                    'LUMO_eV': to_sig4(lumo_ev),
                    'GAP_eV': to_sig4(gap_ev),
                }
                if convert_smiles_flag:
                    ham_info.update({'SMILES': smiles})

                all_ham_info.append(ham_info)

                with open('ham_info.json', 'w') as f:
                    json.dump(ham_info, fp=f, indent=4)

            except Exception as e:
                print(f"Hamiltonian processing failed at idx {idx}: {e}")
                traceback.print_exc()

    end_time = time.time()
    os.chdir(cwd_)

    if all_ham_info:
        df = pd.DataFrame(all_ham_info)
        output_csv_path = os.path.join(cwd_, 'ham_summary.csv')
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved Hamiltonian summary to {output_csv_path}")

    second_per_item = (end_time - start_time) / max(1, len(all_ham_info))
    print(f'Hamiltonian Post-process Time (s/item): {second_per_item}')