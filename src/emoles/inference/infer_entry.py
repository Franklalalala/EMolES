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
from pyscf.scf.hf import dip_moment

# DPTB imports
from dftio.data import _keys
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only
from dptb.data import AtomicDataset, DataLoader, AtomicData, AtomicDataDict
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.utils.tools import j_loader
from dptb.utils.argcheck import collect_cutoffs

# EMOLES imports
from emoles.utils import (
    matrix_transform,
    get_mo_occ,
)
from emoles.inference.common_tools import atom_2_smile, calculate_esp_from_dm, extract_model_params

# ---------------------------------------------------------
# [New/Updated] Imports from emoles.loss & emoles.pyscf
# Replaces local duplicated logic
# ---------------------------------------------------------
from emoles.pyscf import get_dipole_info
from emoles.loss import (
    get_electronic_properties,  # Replaces local definition
    calculate_properties_from_dm,  # For dm_infer_entry
    get_electron_number_from_dm  # For dm_infer_entry
)


# ==========================================
# Helpers
# ==========================================

def _load_npy_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


# ==========================================
# DPTB Inference Logic
# ==========================================

def ase_db_2_dummy_dptb_lmdb(ase_db_path: str, dptb_lmdb_path: str):
    if os.path.exists(dptb_lmdb_path):
        shutil.rmtree(dptb_lmdb_path)

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
                "dielectric_constant": a_row.data.dielectric_constant_weighted_detail.get('dielectric_constant_weighted', 0),
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
    np.save('predicted.npy', ham_ndarray)

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
                           max_items: int = 200, device: str = 'cuda'):
    import e3nn
    e3nn.set_optimization_defaults(jit_script_fx=False)

    device = torch.device(device)
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    basis, r_max = extract_model_params(model)
    print(r_max)
    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)

    # if os.path.exists(abs_out_path):
    #     shutil.rmtree(abs_out_path)
    os.makedirs(abs_out_path, exist_ok=True)

    lmdb_path = os.path.join(abs_out_path, 'lmdb')
    npy_path = os.path.join(abs_out_path, 'results')
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
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
    reference_datasets = build_dataset(basis=basis,
                                       r_max=r_max,
                                       train_w_charge=True,
                                       train_w_eps=True,
                                       **reference_info)
    reference_loader = DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)

    start_time = time.time()
    idx = 0
    for idx, a_ref_batch in enumerate(reference_loader):
        if idx >= max_items:
            break
        batch = a_ref_batch.to(device)
        batch = AtomicData.to_AtomicDataDict(batch)
        with torch.no_grad():
            predicted_data = model(batch)
        save_info_2_npy(folder_path=npy_path, idx=idx, batch_info=predicted_data, model=model, device=device,
                        has_overlap=False)

    end_time = time.time()
    print('DPTB inference done.')
    second_per_item = (end_time - start_time) / max(1, min(idx + 1, max_items))
    print(f'DPTB Inference Time (s/item): {second_per_item}')


# ==========================================
# Post-Processing: Density Matrix
# ==========================================

def get_dm_info_from_npy(ase_db_path,
                         npy_folder_path,
                         convert_smiles_flag=False,
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
                    # Using common_tools version as per original logic for this function
                    pred_esp_max, pred_esp_min = calculate_esp_from_dm(mol, pred_dm, "pred", multiwfn_gen_dm_flag)

                mol_dip = dip_moment(mol, pred_dm, unit='DEBYE')
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                to_sig4 = lambda x: float(f"{x:.4g}")

                dm_info = {
                    'Index': idx,
                    'SMILES': smiles if convert_smiles_flag else "",
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
                # Uses imported function from emoles.loss
                props = get_electronic_properties(
                    mol, ham=pred_ham_pyscf, overlap=overlap, dm=None
                )

                # Note: props["HOMO"] etc from loss.py are in Hartree.
                homo_ev = props["HOMO"] * Hartree
                lumo_ev = props["LUMO"] * Hartree
                gap_ev = props["GAP"] * Hartree

                # 5. Draw HOMO / LUMO Cubes
                if idx < max_cube_save:
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


# ==========================================
# New Pure Inference Entry (DM -> Properties)
# ==========================================
def dm_infer_entry(
        abs_ase_path,
        results_folder_path,
        dm_filename="predicted.npy",
        convention="def2svp",
        mol_charge=0,
        transform_dm_flag=True,
        calc_esp_flag=True,
        calc_electronic_flag=True,
        save_cube_info=True,
        n_save_cube_items=5,
        cube_grid=75,  # Grid resolution for cube generation
        temp_cube_file="infer_cube_data.pkl",
        summary_filename="inference_summary.npz",
        gen_esp_cube_flag: bool = False,
        max_items=None,
        unified_pcm_flag: bool = True,  # 新增 flag: 默认开启，统一使用 PCM，不再分离 gas 和 pcm
):
    """
    Pure inference entry point.
    Loads geometry and predicted DM, calculates properties (Dipole, HOMO/LUMO, ESP),
    and saves results without calculating loss/metrics against a ground truth.

    Refactor:
    - Reuse a shared PySCF mf object within each item (electronic + ESP share the same mf/fock/ovlp/mo_*).
    - dielectric_constant from db is treated as pre-made eps; mf is initialized with this eps and printed.
    """
    import os
    import time
    import json
    import pickle
    import traceback
    import numpy as np
    import pyscf
    from pyscf import dft, tools
    from ase.db import connect
    from tqdm import tqdm
    from ase.units import Hartree

    # ---- Convention / basis ----
    if convention == "6311gdp":
        basis = "6-311+g(d,p)"
        back_convention = "back_2_thu_pyscf"
    else:
        basis = "def2svp"
        back_convention = "back2pyscf"

    results_folder_path = os.path.abspath(results_folder_path)

    def _as_jsonable(x):
        if isinstance(x, (np.ndarray, np.generic)):
            return x.tolist()
        return x

    try:
        from emoles.loss import build_uff_radii_table
    except Exception:
        build_uff_radii_table = None

    start_time = time.time()
    summary_data_list = []
    temp_cube_data = []

    processed = 0
    failed = 0

    with connect(abs_ase_path) as db:
        total_rows = db.count() if max_items is None else max_items

        for idx, a_row in tqdm(enumerate(db.select()), total=total_rows):
            if max_items is not None and idx >= max_items:
                break

            work_dir = os.path.join(results_folder_path, f"{idx}")
            if not os.path.exists(work_dir):
                continue

            cwd_ = os.getcwd()
            try:
                os.chdir(work_dir)
                processed += 1

                # ----------------------------
                # A) Geometry, charge, eps
                # ----------------------------
                atom_nums = a_row.numbers
                an_atoms = a_row.toatoms()

                current_mol_charge = a_row.data.get("charge", mol_charge)
                current_mol_dielectric_constant = a_row.data.dielectric_constant_weighted_detail.get(
                    'dielectric_constant_weighted', 0)

                print(f"[{idx}] charge = {current_mol_charge}")
                print(f"[{idx}] dielectric_constant (eps) = {current_mol_dielectric_constant}")

                Zsum = int(an_atoms.get_atomic_numbers().sum())
                total_electrons = Zsum - current_mol_charge
                mol_spin = total_electrons % 2

                # ----------------------------
                # B) Build PySCF molecule
                # ----------------------------
                mol = pyscf.gto.Mole()
                mol.charge = current_mol_charge
                mol.spin = mol_spin
                mol.build(
                    verbose=0,
                    atom=[[atom_nums[i], at.position] for i, at in enumerate(an_atoms)],
                    basis=basis,
                    unit="ang",
                )
                overlap = mol.intor("int1e_ovlp")

                # ----------------------------
                # C) Load DM (+ transform)
                # ----------------------------
                load_path = dm_filename if not os.path.isabs(dm_filename) else dm_filename
                pred_dm = _load_npy_safe(load_path)

                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)

                # ----------------------------
                # D) Decoupled mean-fields (Gas for shapes, PCM for energies)
                # ----------------------------
                mf_pcm = dft.RKS(mol)
                mf_pcm.xc = "b3lyp"
                eps = current_mol_dielectric_constant
                if eps is not None and float(eps) > 1.0:
                    mf_pcm = mf_pcm.PCM()
                    mf_pcm.with_solvent.eps = float(eps)
                    mf_pcm.with_solvent.method = "IEF-PCM"
                    if build_uff_radii_table is not None:
                        uff_radii_tb = build_uff_radii_table()
                        mf_pcm.with_solvent.radii_table = 1.1 * uff_radii_tb
                    mf_pcm.with_solvent.lebedev_order = 31

                # 最小侵入式修改：根据 flag 决定是否分离 gas 实例
                if unified_pcm_flag:
                    mf_gas = mf_pcm
                else:
                    mf_gas = dft.RKS(mol)
                    mf_gas.xc = "b3lyp"

                # ----------------------------
                # E) Basic properties
                # ----------------------------
                props = {
                    "idx": idx,
                    "charge": int(current_mol_charge),
                    "spin": int(mol_spin),
                }

                dip_vec = get_dipole_info(mol, pred_dm)  # Debye vector (shape property)
                props["dipole_vector"] = dip_vec
                props["dipole_magnitude"] = float(np.linalg.norm(np.array(dip_vec)))

                Ne_pred = get_electron_number_from_dm(pred_dm, overlap)
                charge_from_dm_infer = Zsum - Ne_pred

                props["Ne_actual"] = float(total_electrons)
                props["Ne_pred"] = float(Ne_pred)
                props["Ne_error"] = float(abs(Ne_pred - total_electrons))
                props["charge_from_dm_infer"] = float(charge_from_dm_infer)

                # ----------------------------
                # F) Electronic structure (Decoupled extraction)
                # ----------------------------
                electronic_info_gas = None
                if calc_electronic_flag:
                    # 总是需要计算 PCM 作为主 reference 用于 energy values
                    electronic_info_pcm = get_electronic_properties(
                        mol, dm=pred_dm, overlap=overlap, pcm_eps=float(eps) if eps else None, mf=mf_pcm
                    )

                    if unified_pcm_flag:
                        electronic_info_gas = electronic_info_pcm
                    else:
                        # Unperturbed (Gas) for physical shapes & coefficients
                        electronic_info_gas = get_electronic_properties(
                            mol, dm=pred_dm, overlap=overlap, mf=mf_gas
                        )

                    props["HOMO"] = float(electronic_info_pcm["HOMO"] * Hartree)
                    props["LUMO"] = float(electronic_info_pcm["LUMO"] * Hartree)
                    props["GAP"] = float(electronic_info_pcm["GAP"] * Hartree)

                    # Cubes
                    if save_cube_info and idx < n_save_cube_items:
                        homo_coeff = electronic_info_gas["HOMO_coefficients"]
                        lumo_coeff = electronic_info_gas["LUMO_coefficients"]
                        tools.cubegen.orbital(mol, "homo.cube", homo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)
                        tools.cubegen.orbital(mol, "lumo.cube", lumo_coeff, nx=cube_grid, ny=cube_grid, nz=cube_grid)
                        tools.cubegen.density(mol, "pred_density.cube", pred_dm, nx=cube_grid, ny=cube_grid,
                                              nz=cube_grid)

                # ----------------------------
                # G) ESP / deformation (Requires GAS mf and shapes!)
                # ----------------------------
                if calc_esp_flag:
                    kwargs = dict(
                        mol=mol,
                        dm=pred_dm,
                        prefix="infer",
                        gen_dm_flag=gen_esp_cube_flag,
                        mf=mf_gas,  # 如果 flag 为 True，此处天然变成统一的 mf_pcm
                        overlap=overlap,
                    )

                    if electronic_info_gas is not None:
                        kwargs.update(
                            fock=electronic_info_gas.get("hamiltonian", None),
                            mo_energy=electronic_info_gas.get("mo_energy", None),
                            mo_coeff=electronic_info_gas.get("mo_coeff", None),
                            mo_occ=electronic_info_gas.get("mo_occ", None),
                        )

                    esp_max, esp_min, phi = calculate_properties_from_dm(**kwargs)
                    props["ESP_max_eV"] = float(esp_max)
                    props["ESP_min_eV"] = float(esp_min)
                    props["Deformation_phi"] = None if phi is None else float(phi)

                # ----------------------------
                # H) Save per-item json
                # ----------------------------
                with open("dm_inference_result.json", "w") as f_json:
                    json.dump({k: _as_jsonable(v) for k, v in props.items()}, f_json, indent=4)

                summary_data_list.append(props)

                if save_cube_info and idx < n_save_cube_items and electronic_info_gas is not None:
                    mol_info = {
                        "atom_nums": [int(x) for x in atom_nums],
                        "atom_coords": [list(at.position) for at in an_atoms],
                        "charge": int(current_mol_charge),
                        "spin": int(mol_spin),
                        "basis": basis,
                        "unit": "ang",
                    }
                    temp_cube_data.append(
                        {
                            "idx": idx,
                            "mol_info": mol_info,
                            "outputs": electronic_info_gas,  # unified下为pcm结果，否则为gas结果
                            "tgt_info": None,
                        }
                    )

            except Exception as e:
                failed += 1
                traceback.print_exc()
                print(f"[dm_infer_entry] idx {idx} failed: {repr(e)}")
            finally:
                os.chdir(cwd_)

    if save_cube_info and temp_cube_file and len(temp_cube_data) > 0:
        save_path = os.path.join(results_folder_path, temp_cube_file)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(temp_cube_data, f)
            print(f"[dm_infer_entry] Saved cube info for {len(temp_cube_data)} items to {save_path}")
        except Exception as e:
            print(f"[dm_infer_entry] Failed to save cube data: {e}")

    if len(summary_data_list) > 0:
        all_keys = set().union(*(d.keys() for d in summary_data_list))
        npz_dict = {}
        for key in all_keys:
            values = []
            for item in summary_data_list:
                v = item.get(key, np.nan)
                if isinstance(v, (list, np.ndarray)):
                    values.append(v)
                elif v is None:
                    values.append(np.nan)
                else:
                    values.append(v)
            try:
                npz_dict[key] = np.array(values)
            except Exception:
                npz_dict[key] = np.array(values, dtype=object)

        np.savez(os.path.join(results_folder_path, summary_filename), **npz_dict)
        print(f"[dm_infer_entry] Summary saved to {summary_filename}")

    total_time = time.time() - start_time
    avg_time = total_time / max(1, processed)

    print(f"[dm_infer_entry] Finished. Processed: {processed}, Failed: {failed}")
    print(f"Total Time: {total_time:.2f}s, Avg: {avg_time:.4f}s/item")

    return summary_data_list