import json
import os
import os.path
import torch
import shutil
import time
from argparse import Namespace
from pprint import pprint

import numpy as np
import py3Dmol
import pyscf
from ase.db.core import connect
from learn_qh9.parse_gau_logs_tools import transform_matrix, generate_molecule_transform_indices
from learn_qh9.datasets import matrix_transform
from learn_qh9.trainer import Trainer
from pyscf import tools, scf
from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


loss_weights = {
    'hamiltonian': 1.0,
    'diagonal_hamiltonian': 1.0,
    'non_diagonal_hamiltonian': 1.0,
    'orbital_energies': 1.0,
    "orbital_coefficients": 1.0,
    'HOMO': 1.0, 'LUMO': 1.0, 'GAP': 1.0,
}


def process_cube_file(cube_path):
    # Read the cube file
    with open(cube_path, 'r') as f:
        cube_data = f.read()

    # Create visualization
    view = py3Dmol.view()
    view.addModel(cube_data, 'cube')
    view.addVolumetricData(cube_data, "cube", {'isoval': -0.03, 'color': "red", 'opacity': 0.75})
    view.addVolumetricData(cube_data, "cube", {'isoval': 0.03, 'color': "blue", 'opacity': 0.75})
    view.setStyle({'stick': {}})
    view.zoomTo()

    # Generate HTML content
    html_content = view._make_html()

    # Create HTML file with the same name as the cube file
    html_path = os.path.splitext(cube_path)[0] + '.html'
    with open(html_path, "w") as out:
        out.write(html_content)

    print(f"Generated HTML for: {cube_path}")


def process_batch_cube_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.cube'):
                cube_path = os.path.join(dirpath, filename)
                process_cube_file(cube_path)


def vec_cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return round(float(np.abs(dot_product / (norm_a * norm_b))), 2)


def orbital_similarity(coeff1, coeff2, overlap_12):
    # coeff1, coeff2 = coeff1.numpy(), coeff2.numpy()
    transformed_coeff1 = np.dot(overlap_12.T, coeff1)
    a_sim = vec_cosine_similarity(transformed_coeff1, coeff2)
    return a_sim


def criterion(outputs, target, names, overlap_12):
    error_dict = {}
    for key in names:
        # if key == 'orbital_coefficients':
        #     "The shape if [batch, total_orb, num_occ_orb]."
        #     error_dict[key] = torch.cosine_similarity(outputs[key], target[key], dim=1).abs().mean()

        if key in ['HOMO_orbital_coefficients', 'LUMO_orbital_coefficients']:
            error_dict[key] = orbital_similarity(coeff1=target[key], coeff2=outputs[key], overlap_12=overlap_12)

            # error_dict[key] = vec_cosine_similarity(outputs[key], target[key])

        # elif key in ['diagonal_hamiltonian', 'non_diagonal_hamiltonian']:
        #     diff_blocks = outputs[key].cpu() - target[key].cpu()
        #     mae_blocks = torch.sum(torch.abs(diff_blocks) * target[f"{key}_mask"], dim=[1, 2])
        #     count_sum_blocks = torch.sum(target[f"{key}_mask"], dim=[1, 2])
        #     if key == 'non_diagonal_hamiltonian':
        #         row = target.edge_index_full[0]
        #         batch = target.batch[row]
        #     else:
        #         batch = target.batch
        #     mae_blocks = scatter_sum(mae_blocks, batch)
        #     count_sum_blocks = scatter_sum(count_sum_blocks, batch)
        #     error_dict[key + '_mae'] = (mae_blocks / count_sum_blocks).mean()
        # else:
        #     diff = torch.tensor(outputs[key] - target[key])
        #     mae = torch.mean(torch.abs(diff))
        #     error_dict[key] = mae
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
    # gau_ham = np.load(os.path.join(gau_path, 'original_ham.npy'))
    if not united_overlap_flag:
        gau_overlap = np.load(os.path.join(gau_path, 'overlap.npy'))
        return gau_ham, gau_overlap
    return gau_ham, None


def prepare_np(overlap_matrix, full_hamiltonian, atom_symbols, transform_ham_flag=False, transform_overlap_flag=False, convention='back2pyscf'):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(full_hamiltonian, atom_symbols, convention=convention)
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(overlap_matrix, atom_symbols, convention=convention)
    return full_hamiltonian, overlap_matrix


def test_with_npy(abs_ase_path, npy_folder_path, gau_npy_folder_path, temp_data_file, united_overlap_flag=False):
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
            mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')
            # mol.build(verbose=0, atom=t, basis='6-311+g(d,p)', unit='ang')
            homo_idx = int(sum(atom_nums) / 2) - 1

            mol_target = pyscf.gto.Mole()
            mol_target.build(verbose=0, atom=t, basis='6-311+g(d,p)', unit='ang')
            overlap_12 = pyscf.gto.intor_cross('int1e_ovlp', mol_target, mol)

            if not united_overlap_flag:
                predicted_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/predicted_overlap.npy'))
                original_overlap = np.load(os.path.join(npy_folder_path, f'{idx}/original_overlap.npy'))
                gau_ham, gau_overlap = load_gaussian_data(idx, gau_npy_folder_path, False)
                original_ham, original_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=original_overlap, full_hamiltonian=original_ham, transform_ham_flag=True, transform_overlap_flag=True)
                predicted_ham, predicted_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=predicted_overlap, full_hamiltonian=predicted_ham, transform_ham_flag=True, transform_overlap_flag=True)

            else:
                gau_ham, gau_overlap = load_gaussian_data(idx, gau_npy_folder_path, False)
                target_overlap = mol.intor("int1e_ovlp")
                _, predicted_overlap, original_overlap = target_overlap, target_overlap, target_overlap
                # gau_overlap = mol_target.intor("int1e_ovlp")
                original_ham, original_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=original_overlap, full_hamiltonian=original_ham, transform_ham_flag=True, transform_overlap_flag=False)
                predicted_ham, predicted_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=predicted_overlap, full_hamiltonian=predicted_ham, transform_ham_flag=True, transform_overlap_flag=False)

            gau_ham, gau_overlap = prepare_np(atom_symbols=atom_nums, overlap_matrix=gau_overlap, full_hamiltonian=gau_ham, transform_ham_flag=False, transform_overlap_flag=False, convention='back_2_thu_pyscf')

            predicted_orbital_energies, predicted_orbital_coefficients = cal_orbital_and_energies(full_hamiltonian=predicted_ham, overlap_matrix=predicted_overlap)
            original_orbital_energies, original_orbital_coefficients = cal_orbital_and_energies(full_hamiltonian=original_ham, overlap_matrix=original_overlap)
            gau_orbital_energies, gau_orbital_coefficients = cal_orbital_and_energies(full_hamiltonian=gau_ham, overlap_matrix=gau_overlap)


            outputs = {
                'HOMO_orbital_coefficients': predicted_orbital_coefficients[:, homo_idx],
                'LUMO_orbital_coefficients': predicted_orbital_coefficients[:, homo_idx+1],
            }

            tgt_info = {
                'HOMO_orbital_coefficients': original_orbital_coefficients[:, homo_idx],
                'LUMO_orbital_coefficients': original_orbital_coefficients[:, homo_idx+1],
            }

            gau_info = {
                'HOMO_orbital_coefficients': gau_orbital_coefficients[:, homo_idx],
                'LUMO_orbital_coefficients': gau_orbital_coefficients[:, homo_idx+1],
            }

            # print(os.getcwd())
            # tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_orbital_coefficients[:, homo_idx], nx=n_grid, ny=n_grid, nz=n_grid)

            # error_dict = criterion(outputs, tgt_info, outputs.keys())
            dptb_label_vs_gau = criterion(tgt_info, gau_info, tgt_info.keys(), overlap_12=overlap_12)
            dptb_pred_vs_gau = criterion(outputs, gau_info, outputs.keys(), overlap_12=overlap_12)

            # Store temporary data for cube file generation
            temp_data.append({
                'mol': mol,
                'gau_mol': mol_target,
                'outputs': outputs,
                'tgt_info': tgt_info,
                'gau_info': gau_info,
                'idx': idx,
                'dptb_pred_vs_gau_HOMO_sim': dptb_pred_vs_gau['HOMO_orbital_coefficients'],
                'dptb_label_vs_gau_HOMO_sim': dptb_label_vs_gau['HOMO_orbital_coefficients'],
                'dptb_pred_vs_gau_LUMO_sim': dptb_pred_vs_gau['LUMO_orbital_coefficients'],
                'dptb_label_vs_gau_LUMO_sim': dptb_label_vs_gau['LUMO_orbital_coefficients'],
            })

            for key in dptb_pred_vs_gau.keys():
                if key in total_error_dict.keys():
                    total_error_dict[key] += dptb_pred_vs_gau[key]
                else:
                    total_error_dict[key] = dptb_pred_vs_gau[key]

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

            # if idx == 10:
            #     break

    for key in total_error_dict.keys():
        if key not in ['total_items', 'dptb_label_vs_gau', 'dptb_pred_vs_gau']:
            total_error_dict[key] = total_error_dict[key] / total_error_dict['total_items']

    for comparison in ['dptb_label_vs_gau', 'dptb_pred_vs_gau']:
        for key in total_error_dict[comparison].keys():
            total_error_dict[comparison][key] = total_error_dict[comparison][key] / total_error_dict['total_items']

    end_time = time.time()
    total_error_dict['second_per_item'] = (end_time - start_time) / total_error_dict['total_items']

    # Save all temporary data in a single npz file
    np.savez(temp_data_file, temp_data=temp_data)

    return total_error_dict


def generate_cube_files(temp_data_file, n_grid, cube_dump_place):
    """Generate cube files for HOMO orbitals and save them in sub-folders named by idx."""

    cwd_ = os.getcwd()
    # Load the saved temporary data
    data = np.load(temp_data_file, allow_pickle=True)
    temp_data = data['temp_data']

    for item in temp_data:
        mol = item['mol']
        gau_mol = item['gau_mol']
        outputs = item['outputs']
        tgt_info = item['tgt_info']
        gau_info = item['gau_info']
        idx = item['idx']
        dptb_pred_vs_gau_HOMO_sim = item['dptb_pred_vs_gau_HOMO_sim']
        dptb_label_vs_gau_HOMO_sim = item['dptb_label_vs_gau_HOMO_sim']


        if idx < 10 or dptb_pred_vs_gau_HOMO_sim < 0.7:
            # Create a sub-folder for each idx inside the cube_dump_place
            sub_folder = os.path.join(cube_dump_place, f'idx_{idx}_pred_sim_{dptb_pred_vs_gau_HOMO_sim:.2g}_label_sim_{dptb_label_vs_gau_HOMO_sim:.2g}')
            os.makedirs(sub_folder, exist_ok=True)
            os.chdir(sub_folder)

            tools.cubegen.orbital(mol, 'dptb_predicted_HOMO.cube', outputs['HOMO_orbital_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'dptb_label_HOMO.cube', tgt_info['HOMO_orbital_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(gau_mol, 'gau_HOMO.cube', gau_info['HOMO_orbital_coefficients'], nx=n_grid, ny=n_grid,
                                  nz=n_grid)
            # diff_HOMO = gau_info['HOMO_orbital_coefficients'] - outputs['HOMO_orbital_coefficients']
            # tools.cubegen.orbital(mol, 'gau_prediction_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)
            # diff_HOMO = gau_info['HOMO_orbital_coefficients'] - tgt_info['HOMO_orbital_coefficients']
            # tools.cubegen.orbital(mol, 'gau_label_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)

        os.chdir(cwd_)

if __name__ == '__main__':
    import os
    import json
    from pprint import pprint
    import shutil

    abs_ase_path = r'dump.db'
    npy_folder_path = r'output'
    gau_npy_folder_path = r'/personal/ham_data/1104/no_li/matrix_dump_split/6311gdp/test'
    n_grid = 75

    # Create a temporary folder for storing intermediate results
    os.makedirs(npy_folder_path, exist_ok=True)
    temp_data_file = os.path.abspath('temp_data.npz')
    cube_dump_place = os.path.abspath('cubes')
    if os.path.exists(cube_dump_place):
        shutil.rmtree(cube_dump_place)
    if os.path.exists(temp_data_file):
        os.remove(temp_data_file)
    os.makedirs(cube_dump_place)

    # Call the test_with_npy function (assuming this function is defined elsewhere)
    total_error_dict = test_with_npy(abs_ase_path=abs_ase_path, npy_folder_path=npy_folder_path,
                                     temp_data_file=temp_data_file, gau_npy_folder_path=gau_npy_folder_path, united_overlap_flag=True)

    pprint(total_error_dict)
    with open('test_results.json', 'w') as f:
        json.dump(total_error_dict, f, indent=2)

    # Generate cube files after test_with_npy
    # generate_cube_files(temp_data_file, n_grid, cube_dump_place)

    # Print results

    # Assuming process_batch_cube_file is defined elsewhere
    # process_batch_cube_file(cube_dump_place)
