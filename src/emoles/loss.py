import os
import time

import numpy as np
import pyscf
import torch
from ase.db.core import connect
from ase.units import Hartree
from pyscf.scf.hf import make_rdm1
from tqdm import tqdm
from emoles.pyscf import generate_cube_files, get_dipole_info
from emoles.constant import atom_to_transform_indices
from emoles.utils import cut_and_cal_matrix, format_number, vec_cosine_similarity, get_shifted_ham, get_mo_occ, matrix_transform, generate_molecule_transform_indices


def process_loss_dict(data, item_flag=False):
    # Convert and format values
    processed = {
        'Density-Matrix': data['density_matrix'],
        'Dipole-Moment-magnitude': data['dipole'],
        # Convert to 1e-6 Ha
        'Ham-MAE (1e-6 Ha)': data['hamiltonian'] * 1e6,
        'Diag-MAE (1e-6 Ha)': data['diagonal_hamiltonian_mae'] * 1e6,
        'NonDiag-MAE (1e-6 Ha)': data['non_diagonal_hamiltonian_mae'] * 1e6,

        'Shifted-Ham-MAE (1e-6 Ha)': data['shifted_ham'] * 1e6,
        'Shifted-Diag-MAE (1e-6 Ha)': data['shifted_diagonal_hamiltonian_mae'] * 1e6,
        'Shifted-NonDiag-MAE (1e-6 Ha)': data['shifted_non_diagonal_hamiltonian_mae'] * 1e6,

        'occ-orb-MAE (1e-6 Ha)': data['occupied_orbital_energy'] * 1e6,
        # 'ϵ (1e-6 Ha)': data['orbital_energies'] * 1e6,

        # Convert to percentage
        'occ-orb-Sim (%)': data['orbital_coefficients'] * 1e2,
        # 'ψ (%)': data['orbital_coefficients'] * 1e2,
        'HOMO-Sim (%)': data['HOMO_coefficients'] * 1e2,
        'LUMO-Sim (%)': data['LUMO_coefficients'] * 1e2,

        # Keep in eV
        'HOMO (eV)': data['HOMO'],
        'LUMO (eV)': data['LUMO'],
        'GAP (eV)': data['GAP'],
    }
    if item_flag:
        processed['Time (s/item)'] = data['second_per_item']
        processed['Total Items'] = int(data['total_items'])
        # Format numbers according to the rules
    for key, value in processed.items():
        if key != 'Total Items':  # Skip integer values
            processed[key] = float(format_number(value))

    return processed


def criterion(outputs, target, names, flag=None, atoms=None, mol=None):
    error_dict = {}
    for key in names:
        if key == 'orbital_coefficients':
            output_orbital_coefficients, target_orbital_coefficients = torch.from_numpy(
                outputs[key]).T, torch.from_numpy(target[key]).T
            # output_orbital_coefficients, target_orbital_coefficients = torch.from_numpy(outputs[key]), torch.from_numpy(target[key])
            aa = torch.cosine_similarity(output_orbital_coefficients, target_orbital_coefficients).abs().numpy()
            error_dict[key] = torch.cosine_similarity(output_orbital_coefficients,
                                                      target_orbital_coefficients).abs().mean().numpy()
            if flag:
                print(output_orbital_coefficients.shape)
                print(target_orbital_coefficients.shape)
                print(aa)
                print(error_dict[key])
                # raise RuntimeError
        elif key in ['LUMO_coefficients', 'HOMO_coefficients']:
            error_dict[key] = vec_cosine_similarity(outputs[key], target[key])
            if flag:
                print(error_dict[key])
        elif key == 'density_matrix':
            dm_output = make_rdm1(mo_coeff=outputs[key], mo_occ=outputs['mo_occ'])
            dm_target = make_rdm1(mo_coeff=target[key], mo_occ=outputs['mo_occ'])
            if mol:
                dip_output = get_dipole_info(mol, dm_output)
                dip_target = get_dipole_info(mol, dm_target)
                error_dict['dipole'] = np.abs(np.array(dip_output - dip_target))
            diff_matrix = np.abs(np.array(dm_output - dm_target))
            error_dict[key] = np.mean(diff_matrix)
        elif key == 'shifted_ham':
            diff_matrix = np.abs(np.array(outputs[key] - target[key]))
            error_dict[key] = np.mean(diff_matrix)

            if atoms:
                _, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=atoms.symbols,
                                                                            atom_to_transform_indices=atom_to_transform_indices)
                error_dict['shifted_diagonal_hamiltonian_mae'], error_dict[
                    'shifted_non_diagonal_hamiltonian_mae'] = cut_and_cal_matrix(full_matrix=diff_matrix,
                                                                                 atom_in_mo_indices=atom_in_mo_indices)
        elif key == 'hamiltonian':
            diff_matrix = np.abs(np.array(outputs[key] - target[key]))
            error_dict[key] = np.mean(diff_matrix)
            if atoms:
                _, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=atoms.symbols,
                                                                            atom_to_transform_indices=atom_to_transform_indices)
                error_dict['diagonal_hamiltonian_mae'], error_dict['non_diagonal_hamiltonian_mae'] = cut_and_cal_matrix(
                    full_matrix=diff_matrix[0], atom_in_mo_indices=atom_in_mo_indices)
        else:
            diff = np.array(outputs[key] - target[key])
            mae = np.mean(np.abs(diff))

            if key in ['HOMO', 'LUMO', 'GAP']:
                mae = mae * Hartree

            error_dict[key] = mae
            if flag:
                print(key)
                print(error_dict[key])
    # print(error_dict)
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def post_processing(batch, default_type=np.float32):
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray) and np.issubdtype(batch[key].dtype, np.floating):
            batch[key] = batch[key].astype(default_type)
    return batch


def load_gaussian_data(idx, gau_npy_folder_path, united_overlap_flag):
    gau_path = os.path.join(gau_npy_folder_path, f'{idx}')
    gau_ham = np.load(os.path.join(gau_path, 'fock.npy'))
    # gau_ham = np.load(os.path.join(gau_path, 'original_ham.npy'))[0]
    if not united_overlap_flag:
        gau_overlap = np.load(os.path.join(gau_path, 'overlap.npy'))
        return gau_ham, gau_overlap
    return gau_ham, None


def prepare_np(overlap_matrix, full_hamiltonian, atom_symbols, transform_ham_flag=False, transform_overlap_flag=False,
               convention='def2svp'):
    if convention == '6311gdp':
        back_convention = 'back_2_thu_pyscf'
    else:
        back_convention = 'back2pyscf'

    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(full_hamiltonian, atom_symbols, convention=back_convention)
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(overlap_matrix, atom_symbols, convention=back_convention)
    return full_hamiltonian, overlap_matrix


def test_with_npy(abs_ase_path, npy_folder_path, gau_npy_folder_path, temp_data_file, united_overlap_flag=False,
                  convention='def2svp', mol_charge=0):
    if convention == '6311gdp':
        basis = '6-311+g(d,p)'
    else:
        basis = 'def2svp'

    total_error_dict = {'total_items': 0, 'dptb_label_vs_gau': {}, 'dptb_pred_vs_gau': {}}
    start_time = time.time()

    temp_data = []

    with connect(abs_ase_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            atom_nums = a_row.numbers
            an_atoms = a_row.toatoms()
            total_error_dict['total_items'] += 1

            predicted_ham = np.load(os.path.join(npy_folder_path, f'{idx}/predicted_ham.npy'))
            original_ham = np.load(os.path.join(npy_folder_path, f'{idx}/original_ham.npy'))

            mol = pyscf.gto.Mole()
            t = [[atom_nums[atom_idx], an_atom.position]
                 for atom_idx, an_atom in enumerate(an_atoms)]
            mol.charge = mol_charge
            mol.build(verbose=0, atom=t, basis=basis, unit='ang')
            homo_idx = int((sum(atom_nums) - mol_charge) / 2) - 1
            # print('======================')
            # print('======================')
            # print(atom_nums)
            # print(homo_idx)

            if not united_overlap_flag:
                predicted_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/predicted_overlap.npy'))
                original_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/original_overlap.npy'))
                gau_ham, gau_overlap = load_gaussian_data(idx, gau_npy_folder_path, united_overlap_flag)
                original_ham, original_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=original_overlap,
                                                            full_hamiltonian=original_ham, transform_ham_flag=True,
                                                            transform_overlap_flag=True, convention=convention)
                predicted_ham, predicted_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=predicted_overlap,
                                                              full_hamiltonian=predicted_ham, transform_ham_flag=True,
                                                              transform_overlap_flag=True, convention=convention)
            else:
                gau_ham, _ = load_gaussian_data(idx, gau_npy_folder_path, united_overlap_flag)
                target_overlap = mol.intor("int1e_ovlp")
                gau_overlap, predicted_overlap, original_overlap = target_overlap, target_overlap, target_overlap
                #########################################################################################################################
                if convention == '6311gdp':
                    back_convention = 'back_2_thu_pyscf'
                else:
                    back_convention = 'back2pyscf'
                predicted_ham = matrix_transform(predicted_ham, atom_nums, convention=back_convention)
                mo_occ = get_mo_occ(full_len=predicted_ham.shape[-1], occ_len=homo_idx + 1)

                shifted_label_ham = get_shifted_ham(predicted_ham=predicted_ham, label_ham=gau_ham,
                                                    overlap=original_overlap)
                #########################################################################################################################

                original_ham, original_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=original_overlap,
                                                            full_hamiltonian=original_ham, transform_ham_flag=True,
                                                            transform_overlap_flag=False, convention=convention)
                predicted_ham, predicted_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=predicted_overlap,
                                                              full_hamiltonian=predicted_ham, transform_ham_flag=False,
                                                              transform_overlap_flag=False, convention=convention)

            gau_ham, gau_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=gau_overlap,
                                              full_hamiltonian=gau_ham, transform_ham_flag=False,
                                              transform_overlap_flag=False, convention=convention)
            # print(gau_ham.shape)
            # print(gau_overlap.shape)

            predicted_orbital_energies, predicted_orbital_coefficients = cal_orbital_and_energies(
                full_hamiltonian=predicted_ham, overlap_matrix=predicted_overlap)
            original_orbital_energies, original_orbital_coefficients = cal_orbital_and_energies(
                full_hamiltonian=original_ham, overlap_matrix=original_overlap)
            gau_orbital_energies, gau_orbital_coefficients = cal_orbital_and_energies(full_hamiltonian=gau_ham,
                                                                                      overlap_matrix=gau_overlap)

            pred_HOMO, pred_LUMO = predicted_orbital_energies[homo_idx], predicted_orbital_energies[homo_idx + 1]
            tgt_HOMO, tgt_LUMO = original_orbital_energies[homo_idx], original_orbital_energies[homo_idx + 1]
            gau_HOMO, gau_LUMO = gau_orbital_energies[homo_idx], gau_orbital_energies[homo_idx + 1]

            outputs = {
                'HOMO': pred_HOMO, 'LUMO': pred_LUMO, 'GAP': pred_LUMO - pred_HOMO,
                'hamiltonian': predicted_ham, 'shifted_ham': predicted_ham[0], 'overlap': predicted_overlap,
                'density_matrix': predicted_orbital_coefficients, 'mo_occ': mo_occ,
                'orbital_coefficients': predicted_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_coefficients': predicted_orbital_coefficients[:, homo_idx],
                'LUMO_coefficients': predicted_orbital_coefficients[:, homo_idx + 1],
                'occupied_orbital_energy': predicted_orbital_energies[: homo_idx + 1]
            }

            tgt_info = {
                'HOMO': tgt_HOMO, 'LUMO': tgt_LUMO, 'GAP': tgt_LUMO - tgt_HOMO,
                'hamiltonian': original_ham, 'overlap': original_overlap, 'shifted_ham': shifted_label_ham,
                'density_matrix': original_orbital_coefficients, 'mo_occ': mo_occ,
                'orbital_coefficients': original_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_coefficients': original_orbital_coefficients[:, homo_idx],
                'LUMO_coefficients': original_orbital_coefficients[:, homo_idx + 1],
                'occupied_orbital_energy': original_orbital_energies[: homo_idx + 1]
            }

            gau_info = {
                'HOMO': gau_HOMO, 'LUMO': gau_LUMO, 'GAP': gau_LUMO - gau_HOMO,
                'hamiltonian': gau_ham, 'overlap': gau_overlap, 'shifted_ham': shifted_label_ham,
                'density_matrix': gau_orbital_coefficients, 'mo_occ': mo_occ,
                'orbital_coefficients': gau_orbital_coefficients[:, :homo_idx + 1],
                'HOMO_coefficients': gau_orbital_coefficients[:, homo_idx],
                'LUMO_coefficients': gau_orbital_coefficients[:, homo_idx + 1],
                'occupied_orbital_energy': gau_orbital_energies[: homo_idx + 1]
            }

            # # print(os.getcwd())
            # tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_orbital_coefficients[:, homo_idx], nx=n_grid, ny=n_grid,
            #                       nz=n_grid)

            error_dict = criterion(outputs, gau_info, outputs.keys(), flag=False, atoms=an_atoms, mol=mol)
            dptb_label_vs_gau = criterion(tgt_info, gau_info, tgt_info.keys(), flag=False, atoms=an_atoms, mol=mol)
            # dptb_label_vs_gau = criterion(tgt_info, gau_info, tgt_info.keys(), flag=1)
            # dptb_pred_vs_gau = criterion(outputs, gau_info, outputs.keys(), flag=1)
            dptb_pred_vs_gau = criterion(outputs, gau_info, outputs.keys(), flag=False, atoms=an_atoms, mol=mol)

            # Store temporary data for cube file generation
            temp_data.append({
                'mol': mol,
                'outputs': outputs,
                'tgt_info': tgt_info,
                'gau_info': gau_info,
                'idx': idx,
                'dptb_pred_vs_gau_HOMO_sim': dptb_pred_vs_gau['HOMO_coefficients'],
                'dptb_pred_vs_gau': dptb_pred_vs_gau,
                'dptb_label_vs_gau': dptb_label_vs_gau
            })

            for key in error_dict.keys():
                if key in total_error_dict.keys():
                    total_error_dict[key] += error_dict[key]
                else:
                    total_error_dict[key] = error_dict[key]

            for key in dptb_label_vs_gau.keys():
                if key in total_error_dict['dptb_label_vs_gau'].keys():
                    total_error_dict['dptb_label_vs_gau'][key] += dptb_label_vs_gau[key]
                else:
                    total_error_dict['dptb_label_vs_gau'][key] = dptb_label_vs_gau[key]

            for key in dptb_pred_vs_gau.keys():
                if key in total_error_dict['dptb_pred_vs_gau'].keys():
                    total_error_dict['dptb_pred_vs_gau'][key] += dptb_pred_vs_gau[key]
                else:
                    total_error_dict['dptb_pred_vs_gau'][key] = dptb_pred_vs_gau[key]

            # if idx == 1:
            #     break

    for key in total_error_dict.keys():
        if key not in ['total_items', 'dptb_label_vs_gau', 'dptb_pred_vs_gau']:
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']

    for comparison in ['dptb_label_vs_gau', 'dptb_pred_vs_gau']:
        for key in total_error_dict[comparison].keys():
            total_error_dict[comparison][key] = total_error_dict[comparison][key] / total_error_dict['total_items']

    end_time = time.time()
    total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']
    total_error_dict = process_loss_dict(total_error_dict)

    # Save all temporary data in a single npz file
    np.savez(temp_data_file, temp_data=temp_data[:10])

    return total_error_dict


