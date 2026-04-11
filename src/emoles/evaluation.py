import json
import os
import pickle
import time
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pyscf
import torch
from ase.db.core import connect
from ase.io import write
from ase.units import Hartree
from tqdm import tqdm

from emoles.constant import atom_to_transform_indices, convention_dict
from emoles.electronic import (
    build_uff_radii_table,
    calculate_dm_dipole_mae,
    calculate_properties_from_dm,
    get_electron_number_from_dm,
    get_electronic_properties,
    prepare_np,
)
from emoles.inference.shared import (
    build_pyscf_molecule,
    get_row_charge,
    get_row_dielectric_constant,
    get_row_orbital_labels,
    load_npy_safe,
    resolve_basis_and_convention,
)
from emoles.pyscf import get_dipole_info
from emoles.utils import (
    cut_and_cal_matrix,
    format_number,
    generate_molecule_transform_indices,
    get_shifted_ham,
    matrix_transform,
    vec_cosine_similarity,
)


def process_loss_dict(data, item_flag=False, key="pred_vs_label"):
    if key:
        data = data[key]

    processed = {
        "Density-Matrix": data["density_matrix"],
        "Dipole-Moment-magnitude": data["dipole"],
        "Ham-MAE (1e-6 Ha)": data["hamiltonian"] * 1e6,
        "Diag-MAE (1e-6 Ha)": data["diagonal_hamiltonian_mae"] * 1e6,
        "NonDiag-MAE (1e-6 Ha)": data["non_diagonal_hamiltonian_mae"] * 1e6,
        "Shifted-Ham-MAE (1e-6 Ha)": data["shifted_ham"] * 1e6,
        "Shifted-Diag-MAE (1e-6 Ha)": data["shifted_diagonal_hamiltonian_mae"] * 1e6,
        "Shifted-NonDiag-MAE (1e-6 Ha)": data["shifted_non_diagonal_hamiltonian_mae"] * 1e6,
        "occ-orb-MAE (1e-6 Ha)": data["occupied_orbital_energy"] * 1e6,
        "occ-orb-Sim (%)": data["orbital_coefficients"] * 1e2,
        "HOMO-Sim (%)": data["HOMO_coefficients"] * 1e2,
        "LUMO-Sim (%)": data["LUMO_coefficients"] * 1e2,
        "HOMO (eV)": data["HOMO"],
        "LUMO (eV)": data["LUMO"],
        "GAP (eV)": data["GAP"],
    }
    if item_flag:
        processed["Time (s/item)"] = data["second_per_item"]
        processed["Total Items"] = int(data["total_items"])

    for key_name, value in processed.items():
        if key_name != "Total Items":
            processed[key_name] = float(format_number(value))

    return processed


def criterion(outputs, target, names, flag=None, atoms=None, mol=None):
    error_dict = {}
    for key in names:
        if key == "orbital_coefficients":
            output_orbital_coefficients = torch.from_numpy(outputs[key]).T
            target_orbital_coefficients = torch.from_numpy(target[key]).T
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
                print(error_dict[key])
        elif key in ["LUMO_coefficients", "HOMO_coefficients"]:
            error_dict[key] = vec_cosine_similarity(outputs[key], target[key])
            if flag:
                print(error_dict[key])
        elif key == "density_matrix":
            dm_output = pyscf.scf.hf.make_rdm1(
                mo_coeff=outputs[key], mo_occ=outputs["mo_occ"]
            )
            dm_target = pyscf.scf.hf.make_rdm1(
                mo_coeff=target[key], mo_occ=outputs["mo_occ"]
            )
            if mol:
                dip_output = get_dipole_info(mol, dm_output)
                dip_target = get_dipole_info(mol, dm_target)
                error_dict["dipole"] = np.abs(np.array(dip_output - dip_target))
            error_dict[key] = np.mean(np.abs(np.array(dm_output - dm_target)))
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
                    full_matrix=diff_matrix[0], atom_in_mo_indices=atom_in_mo_indices
                )
        else:
            diff = np.array(outputs[key] - target[key])
            mae = np.mean(np.abs(diff))
            if key in ["HOMO", "LUMO", "GAP"]:
                raw_pred = float(outputs[key])
                raw_label = float(target[key])
                print(f"[{key} DEBUG]")
                print(f"  Pred : {raw_pred:.6f} Ha  =>  {raw_pred * Hartree:.6f} eV")
                print(f"  Label: {raw_label:.6f} Ha  =>  {raw_label * Hartree:.6f} eV")
                print(
                    f"  Diff : {abs(raw_pred - raw_label):.6f} Ha  =>  "
                    f"{abs(raw_pred - raw_label) * Hartree:.6f} eV"
                )
                mae = mae * Hartree

            error_dict[key] = mae
            if flag:
                print(key)
                print(error_dict[key])
    return error_dict


def post_processing(batch, default_type=np.float32):
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray) and np.issubdtype(batch[key].dtype, np.floating):
            batch[key] = batch[key].astype(default_type)
    return batch


def load_gaussian_data(idx, gau_npy_folder_path, united_overlap_flag):
    gau_path = os.path.join(gau_npy_folder_path, f"{idx}")
    gau_ham = np.load(os.path.join(gau_path, "fock.npy"))
    if not united_overlap_flag:
        gau_overlap = np.load(os.path.join(gau_path, "overlap.npy"))
        return gau_ham, gau_overlap
    return gau_ham, None


def process_dm_loss_dict(data, key="pred_vs_label"):
    if key and key in data:
        data = data[key]

    processed = {}
    if "density_matrix" in data:
        processed["Density-Matrix-MAE"] = data["density_matrix"]
    if "diagonal_density_matrix_mae" in data:
        processed["Diag-DM-MAE"] = data["diagonal_density_matrix_mae"]
    if "non_diagonal_density_matrix_mae" in data:
        processed["NonDiag-DM-MAE"] = data["non_diagonal_density_matrix_mae"]
    if "dipole" in data:
        processed["Dipole-Moment-MAE-Debye"] = data["dipole"]
    if "pred_electron_number_error" in data:
        processed["Pred-Ne-Error"] = data["pred_electron_number_error"]
    if "target_electron_number_error" in data:
        processed["Label-Ne-Error"] = data["target_electron_number_error"]
    if "electron_number_pred_vs_target_error" in data:
        processed["Pred-vs-Label-Ne-Error"] = data["electron_number_pred_vs_target_error"]

    orbital_keys = {
        "HOMO": "HOMO-MAE-eV",
        "LUMO": "LUMO-MAE-eV",
        "GAP": "GAP-MAE-eV",
        "pyscf_HOMO": "PySCF-vs-Gaussian-HOMO-MAE-eV",
        "pyscf_LUMO": "PySCF-vs-Gaussian-LUMO-MAE-eV",
        "pyscf_GAP": "PySCF-vs-Gaussian-GAP-MAE-eV",
        "ai_pyscf_HOMO": "AI-vs-PySCF-HOMO-MAE-eV",
        "ai_pyscf_LUMO": "AI-vs-PySCF-LUMO-MAE-eV",
        "ai_pyscf_GAP": "AI-vs-PySCF-GAP-MAE-eV",
        "hamiltonian": "Ham-MAE",
        "diagonal_hamiltonian_mae": "Diag-Ham-MAE",
        "non_diagonal_hamiltonian_mae": "NonDiag-Ham-MAE",
        "orbital_coefficients": "Occupied-Orbital-Sim",
        "HOMO_coefficients": "HOMO-Sim",
        "LUMO_coefficients": "LUMO-Sim",
        "esp_max_mae": "ESP-Max-MAE-eV",
        "esp_min_mae": "ESP-Min-MAE-eV",
        "deformation_factor_mae": "Deformation-Factor-MAE",
    }
    for raw_key, pretty_key in orbital_keys.items():
        if raw_key in data and data[raw_key] is not None:
            processed[pretty_key] = data[raw_key]

    if "second_per_item" in data:
        processed["Time (s/item)"] = data["second_per_item"]

    try:
        for key_name, value in processed.items():
            if key_name not in ["Total Items", "Attempted Items", "Failed Items"]:
                processed[key_name] = float(format_number(value))
    except NameError:
        pass

    return processed


def find_best_dm_transform_permutation(
    abs_ase_path,
    npy_folder_path,
    dm_filename="predicted_dm.npy",
    basis_set="def2svp",
    n_test_items=5,
    base_convention="back2pyscf",
):
    from pyscf import gto

    p_perms = list(permutations([0, 1, 2]))
    d_perms = list(permutations([0, 1, 2, 3, 4]))
    permutation_errors = {(p, d): 0.0 for p in p_perms for d in d_perms}

    print("--- Starting Permutation Search ---")
    print(f"Testing {len(p_perms) * len(d_perms)} combinations on {n_test_items} molecules...")

    valid_items = 0
    with connect(abs_ase_path) as db:
        for idx, row in tqdm(enumerate(db.select()), total=n_test_items):
            if valid_items >= n_test_items:
                break

            folder = os.path.join(npy_folder_path, str(idx))
            dm_path = os.path.join(folder, dm_filename)
            if not os.path.exists(dm_path):
                continue

            try:
                orig_dm = np.load(dm_path)
                atom_nums = row.numbers
                coords = row.toatoms().positions
                charge = row.data.get("charge", 0)

                mol = gto.M(
                    atom=[(atom_nums[i], coords[i]) for i in range(len(atom_nums))],
                    basis=basis_set,
                    charge=charge,
                    spin=(sum(atom_nums) - charge) % 2,
                    unit="Ang",
                    verbose=0,
                )
                overlap = mol.intor("int1e_ovlp")
                target_ne = float(mol.nelectron)

                template_conf = convention_dict[base_convention]
                temp_key = "temp_perm_search"

                for p_idx in p_perms:
                    for d_idx in d_perms:
                        convention_dict[temp_key] = SimpleNamespace(
                            atom_to_orbitals_map=template_conf.atom_to_orbitals_map,
                            orbital_sign_map=template_conf.orbital_sign_map,
                            orbital_order_map=template_conf.orbital_order_map,
                            orbital_idx_map={"s": [0], "p": list(p_idx), "d": list(d_idx)},
                        )
                        transformed_dm = matrix_transform(orig_dm, atom_nums, convention=temp_key)
                        if transformed_dm.ndim == 3:
                            dm_2d = transformed_dm[0]
                            if transformed_dm.shape[0] == 2:
                                dm_2d = np.sum(transformed_dm, axis=0)
                        else:
                            dm_2d = transformed_dm

                        ne_calc = np.einsum("ij,ji->", dm_2d, overlap)
                        permutation_errors[(p_idx, d_idx)] += abs(ne_calc - target_ne)

                valid_items += 1
            except Exception as exc:
                print(f"Skipping idx {idx} due to error: {exc}")
                continue

    if valid_items == 0:
        print("No valid items processed.")
        return

    best_combo = min(permutation_errors, key=permutation_errors.get)
    best_p, best_d = best_combo
    min_error = permutation_errors[best_combo] / valid_items

    print("\n" + "=" * 50)
    print("  SEARCH COMPLETE")
    print("=" * 50)
    print(f"Best Avg Electron Error: {min_error:.2e}")
    print(f"Best P-permutation: {list(best_p)}")
    print(f"Best D-permutation: {list(best_d)}")
    print("-" * 50)
    print("Suggested Convention Dict Entry:\n")
    print("'best_found_convention': Namespace(")
    print(f"    atom_to_orbitals_map={template_conf.atom_to_orbitals_map},")
    print(f"    orbital_idx_map={{'s': [0], 'p': {list(best_p)}, 'd': {list(best_d)}}},")
    print(f"    orbital_sign_map={template_conf.orbital_sign_map},")
    print(f"    orbital_order_map={template_conf.orbital_order_map}")
    print("),")
    print("=" * 50)

    if "temp_perm_search" in convention_dict:
        del convention_dict["temp_perm_search"]


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
    n_save_cube_items=5,
    temp_data_file="temp_cube_data.pkl",
    max_items=300,
    gen_esp_cube_flag=False,
    summary_filename="evaluation_summary.npz",
    pcm_eps: float = 1,
    verbose_profiling: bool = False,
):
    def _log_time(msg):
        if verbose_profiling:
            print(msg)

    basis, back_convention = resolve_basis_and_convention(convention)
    total_error_dict = {"total_items": 0, "pred_vs_label": {}}
    start_time = time.time()
    fail_count = 0
    attempted_count = 0
    failed_indices = []
    summary_data_list = []
    temp_cube_data = []

    global_uff_radii_tb = build_uff_radii_table() if get_ham_flag else None

    with connect(abs_ase_path) as db:
        for idx, row in tqdm(enumerate(db.select())):
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

                atom_nums = row.numbers
                an_atoms = row.toatoms()

                pred_dm = load_npy_safe(pred_dm_filename)
                target_dm = load_npy_safe(target_dm_filename)
                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)
                    target_dm = matrix_transform(target_dm, atom_nums, convention=back_convention)

                t_load = time.time()
                _log_time(f"[{idx}] [Time] NPY Load & Matrix Transform: {t_load - t_item_start:.4f} s")

                current_mol_charge = get_row_charge(row, mol_charge)
                mol, total_electrons, mol_spin = build_pyscf_molecule(
                    an_atoms, basis=basis, charge=current_mol_charge, atom_nums=atom_nums
                )
                overlap = mol.intor("int1e_ovlp")

                t_mol = time.time()
                _log_time(f"[{idx}] [Time] PySCF Mole Build & Overlap: {t_mol - t_load:.4f} s")

                expected_electrons = float(total_electrons)
                ne_pred = get_electron_number_from_dm(pred_dm, overlap)
                ne_target = get_electron_number_from_dm(target_dm, overlap)

                errors = calculate_dm_dipole_mae(pred_dm, target_dm, mol)
                _, atom_in_mo_indices = generate_molecule_transform_indices(
                    atom_types=an_atoms.get_chemical_symbols(),
                    atom_to_transform_indices=atom_to_transform_indices,
                )
                dm_diff = np.abs(pred_dm - target_dm)
                dm_diag, dm_non_diag = cut_and_cal_matrix(
                    full_matrix=dm_diff, atom_in_mo_indices=atom_in_mo_indices
                )
                errors["diagonal_density_matrix_mae"] = dm_diag
                errors["non_diagonal_density_matrix_mae"] = dm_non_diag
                errors["pred_electron_number_error"] = abs(ne_pred - expected_electrons)
                errors["target_electron_number_error"] = abs(ne_target - expected_electrons)
                errors["electron_number_pred_vs_target_error"] = abs(ne_pred - ne_target)

                t_basic_err = time.time()
                _log_time(f"[{idx}] [Time] Basic DM & Dipole Metrics:  {t_basic_err - t_mol:.4f} s")

                electronic_properties_eV = None
                mf_gas = None
                pred_props_gas = None
                target_props_gas = None

                if get_ham_flag:
                    raw_eps = get_row_dielectric_constant(row, pcm_eps)
                    current_pcm_eps = float(raw_eps) if raw_eps is not None else 1.0
                    if current_pcm_eps < 1.0:
                        current_pcm_eps = 1.0

                    mf_gas = pyscf.dft.RKS(mol)
                    mf_gas.xc = "b3lyp"
                    pred_props_gas = get_electronic_properties(mol, dm=pred_dm, overlap=overlap, mf=mf_gas)
                    target_props_gas = get_electronic_properties(mol, dm=target_dm, overlap=overlap, mf=mf_gas)

                    t_mf = time.time()
                    _log_time(f"[{idx}] [Time] Init & Calc Gas Props: {t_mf - t_basic_err:.4f} s")

                    if current_pcm_eps > 1.0:
                        mf_pcm = pyscf.dft.RKS(mol)
                        mf_pcm.xc = "b3lyp"
                        mf_pcm = mf_pcm.PCM()
                        mf_pcm.with_solvent.eps = current_pcm_eps
                        mf_pcm.with_solvent.method = "IEF-PCM"
                        mf_pcm.with_solvent.radii_table = 1.1 * global_uff_radii_tb
                        mf_pcm.with_solvent.lebedev_order = 31
                        pred_props_pcm = get_electronic_properties(mol, dm=pred_dm, overlap=overlap, mf=mf_pcm)
                        target_props_pcm = get_electronic_properties(mol, dm=target_dm, overlap=overlap, mf=mf_pcm)
                    else:
                        pred_props_pcm = pred_props_gas
                        target_props_pcm = target_props_gas

                    t_prop = time.time()
                    _log_time(f"[{idx}] [Time] Check & Calc PCM Props:     {t_prop - t_mf:.4f} s")

                    gaussian_homo, gaussian_lumo, gaussian_gap = get_row_orbital_labels(row, 0.0)

                    pred_homo_ev = float(pred_props_pcm["HOMO"]) * Hartree
                    pred_lumo_ev = float(pred_props_pcm["LUMO"]) * Hartree
                    pred_gap_ev = float(pred_props_pcm["GAP"]) * Hartree
                    pyscf_homo_ev = float(target_props_pcm["HOMO"]) * Hartree
                    pyscf_lumo_ev = float(target_props_pcm["LUMO"]) * Hartree
                    pyscf_gap_ev = float(target_props_pcm["GAP"]) * Hartree

                    electronic_properties_eV = {
                        "pred": {"HOMO_eV": pred_homo_ev, "LUMO_eV": pred_lumo_ev, "GAP_eV": pred_gap_ev},
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

                    ham_orb_errors = criterion(
                        pred_props_gas,
                        target_props_gas,
                        ["hamiltonian", "orbital_coefficients", "HOMO_coefficients", "LUMO_coefficients"],
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
                        temp_cube_data.append(
                            {
                                "idx": idx,
                                "HOMO_sim": errors.get("HOMO_coefficients", 0.0),
                                "mol_info": mol_info,
                                "outputs": pred_props_gas,
                                "tgt_info": target_props_gas,
                            }
                        )
                else:
                    t_crit = time.time()
                    current_pcm_eps = pcm_eps

                if get_esp_sta_flag:
                    if get_ham_flag and pred_props_gas is not None and target_props_gas is not None:
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
                    errors["deformation_factor_mae"] = (
                        abs(p_phi - t_phi) if p_phi is not None and t_phi is not None else None
                    )

                t_esp = time.time()
                if get_esp_sta_flag:
                    _log_time(f"[{idx}] [Time] ESP & Deformation Calc:     {t_esp - t_crit:.4f} s")

                if keep_xyz_file:
                    write("atomic_structure.xyz", an_atoms)

                for key, val in errors.items():
                    if val is not None:
                        total_error_dict["pred_vs_label"][key] = total_error_dict["pred_vs_label"].get(key, 0.0) + val
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
                    "dielectric_constant_used": current_pcm_eps,
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

            except Exception as exc:
                fail_count += 1
                failed_indices.append(idx)
                import traceback

                traceback.print_exc()
                print(f"[evaluate_dm_from_npy] idx {idx} failed: {repr(exc)}")
            finally:
                os.chdir(cwd_)

    if temp_data_file and temp_cube_data:
        save_path = os.path.join(npy_folder_path, temp_data_file)
        try:
            with open(save_path, "wb") as f_obj:
                pickle.dump(temp_cube_data, f_obj)
        except Exception as exc:
            print(f"[evaluate_dm_from_npy] Failed to save temp cube data: {exc}")

    n = total_error_dict["total_items"]
    if n > 0:
        for key in list(total_error_dict["pred_vs_label"].keys()):
            total_error_dict["pred_vs_label"][key] /= n

    end_time = time.time()
    total_error_dict["second_per_item"] = (end_time - start_time) / max(1, n)

    if summary_data_list:
        all_keys = set().union(*(item.keys() for item in summary_data_list))
        npz_dict = {key: np.array([item.get(key, np.nan) for item in summary_data_list]) for key in all_keys}
        np.savez(os.path.join(npy_folder_path, summary_filename), **npz_dict)

    final_data = {"pred_vs_label": total_error_dict["pred_vs_label"].copy()}
    final_data["pred_vs_label"]["second_per_item"] = total_error_dict["second_per_item"]

    result_dict = process_dm_loss_dict(final_data, key="pred_vs_label")
    result_dict.update({"Total Items": n, "Attempted Items": attempted_count, "Failed Items": fail_count})
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
    pcm_eps: float = 25.59,
):
    basis, back_convention = resolve_basis_and_convention(convention)

    total_error_dict = {"total_items": 0, "pred_vs_label": {}}
    start_time = time.time()
    temp_data = []

    with connect(abs_ase_path) as db:
        for idx, row in tqdm(enumerate(db.select())):
            atom_nums = row.numbers
            an_atoms = row.toatoms()
            total_error_dict["total_items"] += 1

            pred_ham = load_npy_safe(os.path.join(npy_folder_path, f"{idx}", "predicted_ham.npy"))
            orig_ham = load_npy_safe(os.path.join(npy_folder_path, f"{idx}", "original_ham.npy"))

            current_mol_charge = get_row_charge(row, mol_charge)
            mol, _, mol_spin = build_pyscf_molecule(
                an_atoms, basis=basis, charge=current_mol_charge, atom_nums=atom_nums
            )

            shifted_label_ham = None
            if not united_overlap_flag:
                pred_ov = load_npy_safe(os.path.join(npy_folder_path, f"{idx}", "predicted_overlap.npy"))
                orig_ov = load_npy_safe(os.path.join(npy_folder_path, f"{idx}", "original_overlap.npy"))
                orig_ham_prep, orig_ov_prep = prepare_np(
                    overlap_matrix=orig_ov,
                    full_hamiltonian=orig_ham,
                    atom_symbols=atom_nums,
                    transform_ham_flag=True,
                    transform_overlap_flag=True,
                    convention=convention,
                )
                pred_ham_prep, pred_ov_prep = prepare_np(
                    overlap_matrix=pred_ov,
                    full_hamiltonian=pred_ham,
                    atom_symbols=atom_nums,
                    transform_ham_flag=True,
                    transform_overlap_flag=True,
                    convention=convention,
                )
            else:
                orig_ham_bt = matrix_transform(orig_ham, atom_nums, convention=back_convention)
                pred_ham_bt = matrix_transform(pred_ham, atom_nums, convention=back_convention)
                target_overlap = mol.intor("int1e_ovlp")

                shifted_label_ham = get_shifted_ham(
                    predicted_ham=pred_ham_bt, label_ham=orig_ham_bt, overlap=target_overlap
                )
                orig_ham_prep, orig_ov_prep = prepare_np(
                    overlap_matrix=target_overlap,
                    full_hamiltonian=orig_ham_bt,
                    atom_symbols=atom_nums,
                    transform_ham_flag=False,
                    transform_overlap_flag=False,
                    convention=convention,
                )
                pred_ham_prep, pred_ov_prep = prepare_np(
                    overlap_matrix=target_overlap,
                    full_hamiltonian=pred_ham_bt,
                    atom_symbols=atom_nums,
                    transform_ham_flag=False,
                    transform_overlap_flag=False,
                    convention=convention,
                )

            outputs = get_electronic_properties(mol, ham=pred_ham_prep, overlap=pred_ov_prep, pcm_eps=pcm_eps)
            tgt_info = get_electronic_properties(
                mol,
                ham=orig_ham_prep,
                overlap=orig_ov_prep,
                shifted_ham=shifted_label_ham,
                pcm_eps=pcm_eps,
            )

            pred_vs_label = criterion(outputs, tgt_info, list(outputs.keys()), flag=False, atoms=an_atoms, mol=mol)
            for key, value in pred_vs_label.items():
                total_error_dict["pred_vs_label"][key] = total_error_dict["pred_vs_label"].get(key, 0.0) + value

            if save_summary:
                mol_info = {
                    "atom_nums": [int(x) for x in atom_nums],
                    "atom_coords": [list(at.position) for at in an_atoms],
                    "charge": current_mol_charge,
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
        with open(temp_data_file, "wb") as f_obj:
            pickle.dump(temp_data, f_obj)

    return total_error_dict
