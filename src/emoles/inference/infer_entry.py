import os
import shutil
import json

import lmdb
import pickle
import numpy as np
from ase.db import connect
from dftio.data import _keys
import torch
from dptb.nn.hr2hk import HR2HK
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import normalize, collect_cutoffs
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from emoles.utils import matrix_transform
from pyscf import gto, dft, tools
import pyscf
from pyscf.scf.hf import dip_moment
from emoles.inference.common_tools import atom_2_smile, calculate_esp_from_dm
from tqdm import tqdm
import pandas as pd
import time


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
    os.makedirs(f'{idx}')
    os.chdir(f'{idx}')
    batch_info['kpoint'] = torch.tensor([0.0, 0.0, 0.0], device=device)
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
    np.save('predicted.npy', ham_ndarray)

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
        np.save('predicted_overlap.npy', overlap_ndarray[0])
    os.chdir(cwd_)


default_ckpt_path = r'/opt/example/dptb/1105_infer_utils/def2svp_batch_size_1.pth'
default_input_json_path = r'/opt/example/dptb/1105_infer_utils/def2svp_batch_size_1.json'

default_pretrained_ckpt_path = r'/opt/example/dptb/dptb_pretrain_utils/best.pth'
default_pretrained_input_json_path = r'/opt/example/dptb/dptb_pretrain_utils/input.json'

default_fine_tune_ckpt_path = r'/share/dptb_ckpt/fine_tune/best.pth'
default_fine_tune_input_json_path = r'/share/dptb_ckpt/fine_tune/input.json'


def dptb_infer_from_ase_db(ase_db_path: str, out_path: str, checkpoint_path: str=default_fine_tune_ckpt_path, input_json_path: str=default_fine_tune_input_json_path, limit: int=200, device: str='cuda'):
    device = device
    device = torch.device(device)
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)
    if os.path.exists(abs_out_path):
        shutil.rmtree(abs_out_path)
    os.makedirs(abs_out_path)
    lmdb_path = os.path.join(abs_out_path, 'lmdb')
    npy_path = os.path.join(abs_out_path, 'npy')
    os.makedirs(npy_path)
    jdata = j_loader(input_json_path)
    cutoff_options = collect_cutoffs(jdata)
    ase_db_2_dummy_dptb_lmdb(ase_db_path, lmdb_path)
    reference_info = {
        "root": lmdb_path,
        "prefix": "data",
        "type": "LMDBDataset",
        "get_Hamiltonian": False,
        "get_overlap": False
    }
    reference_datasets = build_dataset(**cutoff_options, **reference_info, **jdata["common_options"])
    reference_loader = DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)
    start_time = time.time()
    for idx, a_ref_batch in enumerate(reference_loader):
        batch = a_ref_batch.to(device)
        batch = AtomicData.to_AtomicDataDict(batch)
        with torch.no_grad():
            predicted_data = model(batch)
        save_info_2_npy(folder_path=npy_path, idx=idx, batch_info=predicted_data, model=model, device=device, has_overlap=False)
        if idx == limit - 1:
            break
    end_time = time.time()
    print('DPTB inference done.')
    second_per_item = (end_time - start_time) / min(1 + idx, limit)
    print(f'DPTB Inference Time (s/item): {second_per_item}')


def get_dm_info_from_npy(ase_db_path,
                         npy_folder_path,
                         convention='def2svp',
                         mol_charge=0,
                         pred_dm_filename='predicted.npy',
                         transform_dm_flag=True,
                         get_esp_sta_flag=True,
                         get_dm_cube_flag=False,
                         dm_cube_src='pyscf',
                         keep_xyz_file=True,
                         max_cube_save: int = 5,
                         max_items: int = 300,
                         dm_grid: int = 40,
                         ):
    print('Start postprocess')
    if convention == '6311gdp':
        basis = '6-311+g(d,p)'
        back_convention = 'back_2_thu_pyscf'
    else:
        basis = 'def2svp'
        back_convention = 'back2pyscf'

    def _load_npy_safe(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        return np.load(path)

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()
    all_dm_info = []
    start_time = time.time()
    with connect(ase_db_path) as db:
        for idx, a_row in tqdm(enumerate(db.select())):
            if idx == max_items:
                break
            os.chdir(os.path.join(npy_folder_path, f'{idx}'))

            atom_nums = a_row.numbers
            an_atoms = a_row.toatoms()
            smiles = atom_2_smile(an_atoms)

            pred_dm = _load_npy_safe(pred_dm_filename)
            pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)

            mol = pyscf.gto.Mole()
            t = [[atom_nums[atom_idx], an_atom.position]
                 for atom_idx, an_atom in enumerate(an_atoms)]
            mol.charge = mol_charge
            mol.build(verbose=0, atom=t, basis=basis, unit='ang')

            multiwfn_gen_dm_flag = False
            if get_dm_cube_flag and idx < max_cube_save:
                if dm_cube_src == 'pyscf':
                    tools.cubegen.density(mol, 'pred_electron_density.cube', pred_dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                    tools.cubegen.mep(mol, 'pred_molecular_electrostatic_potential.cube', pred_dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                else:
                    multiwfn_gen_dm_flag = True
            pred_esp_max, pred_esp_min = calculate_esp_from_dm(mol, pred_dm, "pred", multiwfn_gen_dm_flag)
            mol_dip = dip_moment(mol, pred_dm, unit='DEBYE')
            dip_magnitude = np.linalg.norm(np.array(mol_dip))
            to_sig4 = lambda x: float(f"{x:.4g}")

            dm_info = {
                'Index': idx,
                'SMILES': smiles,
                'Dipole-X-Debye': to_sig4(mol_dip[0]),
                'Dipole-Y-Debye': to_sig4(mol_dip[1]),
                'Dipole-Z-Debye': to_sig4(mol_dip[2]),
                # 将列表中的每个元素也转换为 6 位有效数字
                'Dipole-Moment-Debye': [to_sig4(x) for x in mol_dip],
                'Dipole-Moment-magnitude-Debye': to_sig4(dip_magnitude),
                'ESP-Max-eV': to_sig4(pred_esp_max),
                'ESP-Min-eV': to_sig4(pred_esp_min),
            }
            all_dm_info.append(dm_info)
            with open('dm_info.json', 'w') as f:
                json.dump(dm_info, fp=f, indent=4)
    end_time = time.time()
    os.chdir(cwd_)

    if len(all_dm_info) == 0:
        print("⚠️ Warning: all_dm_info is empty.")
    else:
        print(f"Total records collected: {len(all_dm_info)}")
        print(f"Available keys (columns): {list(all_dm_info[0].keys())}")
        print("Sample entry (truncated):")
        for k, v in all_dm_info[0].items():
            if isinstance(v, (list, tuple)):
                print(f"  {k}: list[{len(v)}]")
            elif hasattr(v, 'shape'):
                print(f"  {k}: ndarray{v.shape}")
            else:
                print(f"  {k}: {v}")

    print("\nData collection complete. Converting to DataFrame and saving to CSV...")
    df = pd.DataFrame(all_dm_info)
    output_csv_path = os.path.join(cwd_, 'dm_summary.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved data to {output_csv_path}")
    second_per_item = (end_time - start_time) / min(1 + idx, max_items)
    print(f'Post-process Time (s/item): {second_per_item}')
