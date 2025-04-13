import os
import shutil

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
    a_ham_hr2hk = HR2HK(
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
    np.save('predicted_ham.npy', ham_ndarray[0])

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


def dptb_infer_from_ase_db(ase_db_path: str, out_path: str, checkpoint_path: str=default_fine_tune_ckpt_path, input_path: str=default_fine_tune_input_json_path, limit: int=200, device: str='cuda'):
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
    cube_path = os.path.join(abs_out_path, 'cube')
    os.makedirs(cube_path)
    jdata = j_loader(input_path)
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
    print(len(reference_loader))
    for idx, a_ref_batch in enumerate(reference_loader):
        batch = a_ref_batch.to(device)
        batch = AtomicData.to_AtomicDataDict(batch)
        with torch.no_grad():
            predicted_data = model(batch)
        save_info_2_npy(folder_path=cube_path, idx=idx, batch_info=predicted_data, model=model, device=device, has_overlap=False)
        if idx == limit - 1:
            break

