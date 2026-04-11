import json
import os
import shutil

import numpy as np
from pyscf import tools, scf, gto, dft
from pyscf.scf.hf import dip_moment
from emoles.utils import matrix_transform, get_atom_in_mo_indices, cut_matrix
from emoles.constant import convention_dict
from dptb.data.build import build_dataset
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
from dptb.nn.hr2hk import HR2HK
import lmdb
import torch
from tqdm import tqdm
import pickle

def generate_cube_files_old(temp_data_file, n_grid, cube_dump_place):
    """Generate cube files for HOMO orbitals and save them in sub-folders named by idx."""

    cwd_ = os.getcwd()
    # Load the saved temporary data
    data = np.load(temp_data_file, allow_pickle=True)
    temp_data = data['temp_data']

    for item in temp_data:
        mol = item['mol']
        outputs = item['outputs']
        tgt_info = item['tgt_info']
        gau_info = item['gau_info']
        idx = item['idx']
        dptb_pred_vs_gau_HOMO_sim = item['dptb_pred_vs_gau_HOMO_sim']

        # Create a sub-folder for each idx inside the cube_dump_place
        sub_folder = os.path.join(cube_dump_place, f'idx_{idx}_sim_{dptb_pred_vs_gau_HOMO_sim:.2g}')
        os.makedirs(sub_folder, exist_ok=True)
        os.chdir(sub_folder)
        if idx < 5:
            tools.cubegen.orbital(mol, 'dptb_predicted_HOMO.cube', outputs['HOMO_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'dptb_label_HOMO.cube', tgt_info['HOMO_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_info['HOMO_coefficients'], nx=n_grid, ny=n_grid,
                                  nz=n_grid)
            diff_HOMO = gau_info['HOMO_coefficients'] - outputs['HOMO_coefficients']
            tools.cubegen.orbital(mol, 'gau_prediction_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)
            diff_HOMO = gau_info['HOMO_coefficients'] - tgt_info['HOMO_coefficients']
            tools.cubegen.orbital(mol, 'gau_label_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)

        os.chdir(cwd_)


def generate_cube_files(temp_data_file, n_grid, cube_dump_place, n_max_cubes):
    """
    Generate cube files for predicted vs target HOMO and LUMO orbitals.

    参数：
        temp_data_file (str): pickle 文件路径，由 get_mae_from_npy 保存。
        n_grid (int): cube 网格数。
        cube_dump_place (str): 输出 cube 文件的根目录。
    """
    import pickle
    from pyscf import gto, tools

    cwd_ = os.getcwd()
    with open(temp_data_file, "rb") as f:
        temp_data = pickle.load(f)

    for item in temp_data:
        idx = item.get('idx')
        HOMO_sim = item.get('HOMO_sim', 0.0)

        # Create a sub-folder for each idx inside the cube_dump_place
        sub_folder = os.path.join(cube_dump_place, f'idx_{idx}_HOMO_sim_{HOMO_sim:.2g}')
        os.makedirs(sub_folder, exist_ok=True)
        os.chdir(sub_folder)

        # 只有当有 mol_info 且有 outputs/tgt_info 时才生成 cube
        if idx < n_max_cubes and ('mol_info' in item) and ('outputs' in item) and ('tgt_info' in item):
            mol_info = item['mol_info']
            # rebuild mol
            mol = gto.Mole()
            atom_list = []
            atom_nums = mol_info['atom_nums']
            atom_coords = mol_info['atom_coords']
            for z, coord in zip(atom_nums, atom_coords):
                atom_list.append([int(z), tuple(map(float, coord))])
            mol.charge = mol_info.get('charge', 0)
            mol.spin = mol_info.get('spin', 0)
            print(mol.spin)
            print(mol.charge)

            mol.build(verbose=0, atom=atom_list, basis=mol_info.get('basis', 'def2svp'), unit=mol_info.get('unit', 'ang'))

            outputs = item['outputs']  # pred 信息
            tgt_info = item['tgt_info']  # label 信息

            # HOMO
            tools.cubegen.orbital(mol, 'pred_HOMO.cube', outputs['HOMO_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'tgt_HOMO.cube', tgt_info['HOMO_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)

            # LUMO
            tools.cubegen.orbital(mol, 'pred_LUMO.cube', outputs['LUMO_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'tgt_LUMO.cube', tgt_info['LUMO_coefficients'], nx=n_grid, ny=n_grid, nz=n_grid)

        # 回到原来的 cwd
        os.chdir(cwd_)


def get_dipole_info(mol, dm):
    mol_dip = dip_moment(mol, dm, unit='DEBYE')
    return np.array(mol_dip, dtype=float)
    # dipole_info = {
    #     'Dipole_Moment_Vector_DEBYE': mol_dip.tolist(),
    #     'Dipole_Moment_Norm_DEBYE': float(dip_magnitude),
    # }
    # return dipole_info


def add_dipole_to_lmdb(cutoff_dict, basis_dict, old_lmdb_path, new_lmdb_path,
                       convention, keep_old_lmdb: bool = True, mol_charge: int=0):
    if convention == '6311gdp':
        pyscf_basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
    else:
        back_convention = 'back2pyscf'
        pyscf_basis = 'def2svp'

    if os.path.exists(new_lmdb_path):
        shutil.rmtree(new_lmdb_path)
    os.makedirs(new_lmdb_path)

    old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
    with old_db_env.begin() as old_txn:
        stat = old_txn.stat()
        entries = stat['entries']
    old_db_env.close()

    root, prefix = os.path.split(old_lmdb_path)
    prefix = prefix.split('.lmdb')[0]
    dataset = build_dataset(
        root=root,
        type="LMDBDataset",
        prefix=prefix,
        get_overlap=True,
        get_Hamiltonian=True,
        basis=basis_dict,
        r_max=cutoff_dict,
    )
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # data_loader = list(data_loader)
    ham_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device='cpu'
    )
    overlap_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
        node_field=AtomicDataDict.NODE_OVERLAP_KEY,
        out_field=AtomicDataDict.OVERLAP_KEY,
        overlap=True,
        device='cpu'
    )
    kpoint = torch.tensor([[0.0, 0.0, 0.0]], device='cpu')


    idx_list = list(range(entries))
    counter = 0
    batch_size = 50
    # Process in batches
    for batch_start in tqdm(range(0, entries, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, entries)
        batch_indices = idx_list[batch_start:batch_end]

        # Open LMDB environments for each batch
        old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
        new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

        with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
            for idx in batch_indices:
                if idx == None:
                    continue
                # Get data from old LMDB
                data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                data_dict = pickle.loads(data_dict)
                ########################################
                ########################################
                # Get dipole
                atomic_numbers = data_dict['atomic_numbers']
                positions = data_dict['pos']
                atom_list = []
                for Z, pos in zip(atomic_numbers, positions):
                    atom_list.append(f"{Z} {pos[0]} {pos[1]} {pos[2]}")
                atom_str = "\n".join(atom_list)
                mol = gto.M(
                    atom=atom_str,
                    basis=pyscf_basis,
                    unit="Angstrom",
                    charge=mol_charge
                )
                mf = dft.RKS(mol)
                mf.xc = 'B3LYP'
                mf.grids.level = 5

                # print(atom_str)
                # mf.disp = 'd3zero'
                ########################################
                # get pyscf-format matrix
                data = dataset[idx]
                data = AtomicData.to_AtomicDataDict(data)
                data['kpoint'] = kpoint
                ham_out_data = ham_hr2hk.forward(data)
                hamiltonian = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
                ham_mat = hamiltonian.real.numpy()
                new_ham_mat = matrix_transform(ham_mat, atomic_numbers, back_convention)

                overlap_out_data = overlap_hr2hk.forward(data)
                overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
                overlap_mat = overlap.real.numpy()
                new_overlap_mat = matrix_transform(overlap_mat, atomic_numbers, back_convention)
                ########################################
                # print(new_ham_mat)
                # print(new_overlap_mat)
                # print('########################')
                # print('########################')
                # print(new_overlap_mat - new_ham_mat)
                # print('########################')
                # print('########################')
                # print(new_ham_mat.shape)
                # print(new_overlap_mat.shape)

                mo_energy, mo_coeff = mf.eig(new_ham_mat[0], new_overlap_mat[0])
                mo_occ = mf.get_occ(mo_energy)
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                dip = mf.dip_moment(dm=dm, unit="Debye", verbose=0)

                data_dict['dipole_moment'] = dip
                data_dict['dipole_magnitude'] = np.linalg.norm(dip)

                ## Debug:
                # print(dip)
                # print(data_dict['dipole_magnitude'])
                ########################################
                # mf.kernel()
                # dm_new = mf.make_rdm1()
                # dip_new = mf.dip_moment(dm=dm_new, unit="Debye")
                # print("Dipole (Debye):", dip_new)
                # print("Dipole magnitude (Debye):", np.linalg.norm(dip_new))
                # print(dm - dm_new)
                # print(dip_new - dip)
                #
                # pyscf_fock = mf.get_fock()
                # np.save('pyscf_fock.npy', pyscf_fock)
                # np.save('lmdb_ham.npy', new_ham_mat[0])
                # pyscf_overlap = mf.get_ovlp()
                # np.save('pyscf_overlap.npy', pyscf_overlap)
                # np.save('lmdb_overlap.npy', new_overlap_mat[0])
                #
                # print(pyscf_fock - new_ham_mat[0])
                # np.save('diff_fock.npy', pyscf_fock - new_ham_mat[0])
                #
                # print(pyscf_overlap - new_overlap_mat[0])
                # np.save('diff_overlap.npy', pyscf_overlap - new_overlap_mat[0])
                #
                # raise RuntimeError
                ########################################

                # Store in new LMDB
                data_dict = pickle.dumps(data_dict)
                new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_dict)
                counter = counter + 1

        # Close LMDB environments after each batch
        old_db_env.close()
        new_db_env.close()

    # Remove old LMDB if requested
    if not keep_old_lmdb:
        shutil.rmtree(old_lmdb_path)



def add_dm_dipole(cutoff_dict, basis_dict, old_lmdb_path, new_lmdb_path,
                       convention, keep_old_lmdb: bool = True, batch_size: int=50, mol_charge: int=0):
    if convention == '6311gdp':
        pyscf_basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
        pyscf_2_dftio_convention = 'pyscf_6311_plus_gdp'
    else:
        back_convention = 'back2pyscf'
        pyscf_basis = 'def2svp'
        pyscf_2_dftio_convention = 'pyscf_def2svp'

    if os.path.exists(new_lmdb_path):
        shutil.rmtree(new_lmdb_path)
    os.makedirs(new_lmdb_path)

    old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
    with old_db_env.begin() as old_txn:
        stat = old_txn.stat()
        entries = stat['entries']
    old_db_env.close()

    root, prefix = os.path.split(old_lmdb_path)
    prefix = prefix.split('.lmdb')[0]
    dataset = build_dataset(
        root=root,
        type="LMDBDataset",
        prefix=prefix,
        get_overlap=True,
        get_Hamiltonian=True,
        basis=basis_dict,
        r_max=cutoff_dict,
    )
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # data_loader = list(data_loader)
    ham_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device='cpu'
    )
    overlap_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
        node_field=AtomicDataDict.NODE_OVERLAP_KEY,
        out_field=AtomicDataDict.OVERLAP_KEY,
        overlap=True,
        device='cpu'
    )
    kpoint = torch.tensor([[0.0, 0.0, 0.0]], device='cpu')

    idx_list = list(range(entries))
    counter = 0
    # Process in batches
    for batch_start in tqdm(range(0, entries, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, entries)
        batch_indices = idx_list[batch_start:batch_end]

        # Open LMDB environments for each batch
        old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
        new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

        with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
            for idx in batch_indices:
                if idx == None:
                    continue
                # Get data from old LMDB
                data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                data_dict = pickle.loads(data_dict)
                ########################################
                ########################################
                # Get dipole
                atomic_numbers = data_dict['atomic_numbers']
                positions = data_dict['pos']
                atom_list = []
                for Z, pos in zip(atomic_numbers, positions):
                    atom_list.append(f"{Z} {pos[0]} {pos[1]} {pos[2]}")
                atom_str = "\n".join(atom_list)
                mol = gto.M(
                    atom=atom_str,
                    basis=pyscf_basis,
                    unit="Angstrom",
                    charge=mol_charge
                )
                mf = dft.RKS(mol)
                mf.xc = 'B3LYP'
                mf.grids.level = 5

                # print(atom_str)
                # mf.disp = 'd3zero'
                ########################################
                # get pyscf-format matrix
                data = dataset[idx]
                data = AtomicData.to_AtomicDataDict(data)
                data['kpoint'] = kpoint
                ham_out_data = ham_hr2hk.forward(data)
                hamiltonian = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
                ham_mat = hamiltonian.real.numpy()
                new_ham_mat = matrix_transform(ham_mat, atomic_numbers, back_convention)

                overlap_out_data = overlap_hr2hk.forward(data)
                overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
                overlap_mat = overlap.real.numpy()
                new_overlap_mat = matrix_transform(overlap_mat, atomic_numbers, back_convention)
                ########################################
                mo_energy, mo_coeff = mf.eig(new_ham_mat[0], new_overlap_mat[0])
                mo_occ = mf.get_occ(mo_energy)
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                dip = mf.dip_moment(dm=dm, unit="Debye", verbose=0)

                data_dict['dipole_moment'] = dip
                data_dict['dipole_magnitude'] = np.linalg.norm(dip)

                dftio_format_dm = matrix_transform(dm, atomic_numbers, pyscf_2_dftio_convention)
                atom_in_mo_indices = get_atom_in_mo_indices(atomic_numbers=atomic_numbers,
                                                            convention_name=pyscf_2_dftio_convention,
                                                            convention_dict=convention_dict)
                cut_dm = cut_matrix(full_matrix=dftio_format_dm, atom_in_mo_indices=atom_in_mo_indices)
                data_dict['density_matrix'] = cut_dm

                # Store in new LMDB
                data_dict = pickle.dumps(data_dict)
                new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_dict)
                counter = counter + 1

        # Close LMDB environments after each batch
        old_db_env.close()
        new_db_env.close()

    # Remove old LMDB if requested
    if not keep_old_lmdb:
        shutil.rmtree(old_lmdb_path)


def add_orb_e_C(cutoff_dict, basis_dict, old_lmdb_path, new_lmdb_path,
                       convention, keep_old_lmdb: bool = True, batch_size: int=50):
    if convention == '6311gdp':
        pyscf_basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
        pyscf_2_dftio_convention = 'pyscf_6311_plus_gdp'
    else:
        back_convention = 'back2pyscf'
        pyscf_basis = 'def2svp'
        pyscf_2_dftio_convention = 'pyscf_def2svp'

    if os.path.exists(new_lmdb_path):
        shutil.rmtree(new_lmdb_path)
    os.makedirs(new_lmdb_path)

    old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
    with old_db_env.begin() as old_txn:
        stat = old_txn.stat()
        entries = stat['entries']
    old_db_env.close()

    root, prefix = os.path.split(old_lmdb_path)
    prefix = prefix.split('.lmdb')[0]
    dataset = build_dataset(
        root=root,
        type="LMDBDataset",
        prefix=prefix,
        get_overlap=True,
        get_Hamiltonian=True,
        basis=basis_dict,
        r_max=cutoff_dict,
    )
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # data_loader = list(data_loader)
    ham_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
        node_field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field=AtomicDataDict.HAMILTONIAN_KEY,
        overlap=True,
        device='cpu'
    )
    overlap_hr2hk = HR2HK(
        idp=dataset.transform,
        edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
        node_field=AtomicDataDict.NODE_OVERLAP_KEY,
        out_field=AtomicDataDict.OVERLAP_KEY,
        overlap=True,
        device='cpu'
    )
    kpoint = torch.tensor([[0.0, 0.0, 0.0]], device='cpu')

    idx_list = list(range(entries))
    counter = 0
    # Process in batches
    for batch_start in tqdm(range(0, entries, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, entries)
        batch_indices = idx_list[batch_start:batch_end]

        # Open LMDB environments for each batch
        old_db_env = lmdb.open(old_lmdb_path, readonly=True, lock=False)
        new_db_env = lmdb.open(new_lmdb_path, map_size=1048576000000, lock=True)

        with old_db_env.begin() as old_txn, new_db_env.begin(write=True) as new_txn:
            for idx in batch_indices:
                if idx == None:
                    continue
                # Get data from old LMDB
                data_dict = old_txn.get(idx.to_bytes(length=4, byteorder='big'))
                data_dict = pickle.loads(data_dict)
                ########################################
                ########################################
                # Get dipole
                atomic_numbers = data_dict['atomic_numbers']
                positions = data_dict['pos']
                atom_list = []
                for Z, pos in zip(atomic_numbers, positions):
                    atom_list.append(f"{Z} {pos[0]} {pos[1]} {pos[2]}")
                atom_str = "\n".join(atom_list)
                mol = gto.M(
                    atom=atom_str,
                    basis=pyscf_basis,
                    unit="Angstrom"
                )
                mf = dft.RKS(mol)
                mf.xc = 'B3LYP'
                mf.grids.level = 5

                # print(atom_str)
                # mf.disp = 'd3zero'
                ########################################
                # get pyscf-format matrix
                data = dataset[idx]
                data = AtomicData.to_AtomicDataDict(data)
                data['kpoint'] = kpoint
                ham_out_data = ham_hr2hk.forward(data)
                hamiltonian = ham_out_data[AtomicDataDict.HAMILTONIAN_KEY]
                ham_mat = hamiltonian.real.numpy()
                new_ham_mat = matrix_transform(ham_mat, atomic_numbers, back_convention)

                overlap_out_data = overlap_hr2hk.forward(data)
                overlap = overlap_out_data[AtomicDataDict.OVERLAP_KEY]
                overlap_mat = overlap.real.numpy()
                new_overlap_mat = matrix_transform(overlap_mat, atomic_numbers, back_convention)

                # pyscf_overlap = mf.get_ovlp()
                # aa = pyscf_overlap - new_overlap_mat[0]
                # print(aa[:20, :20])

                ########################################
                mo_energy, mo_coeff = mf.eig(new_ham_mat[0], new_overlap_mat[0])
                data_dict['orbital_energies'] = mo_energy
                data_dict['orbital_coefficients'] = mo_coeff
                # Store in new LMDB
                data_dict = pickle.dumps(data_dict)
                new_txn.put(counter.to_bytes(length=4, byteorder='big'), data_dict)
                counter = counter + 1

        # Close LMDB environments after each batch
        old_db_env.close()
        new_db_env.close()

    # Remove old LMDB if requested
    if not keep_old_lmdb:
        shutil.rmtree(old_lmdb_path)

