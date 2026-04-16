import json
import os
import pickle
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import pyscf
from ase.db import connect
from ase.io import write
from ase.units import Hartree
from pyscf import dft, tools
from pyscf.scf.hf import dip_moment
from tqdm import tqdm

from emoles.electronic import (
    build_uff_radii_table,
    calculate_esp_from_dm,
    calculate_properties_from_dm,
    get_electronic_properties,
    get_electron_number_from_dm,
)
from emoles.inference.chemistry import atom_2_smile
from emoles.inference.common_tools import (
    build_pyscf_molecule,
    get_row_charge,
    get_row_dielectric_constant,
    load_npy_safe,
    resolve_basis_and_convention,
)
from emoles.inference.fragment_hosting import infer_orbital_fragment_hosting
from emoles.pyscf import get_dipole_info
from emoles.utils import get_pickle_record_any, matrix_transform

DEFAULT_UPDATED_ASE_DB_NAME = "dm_inference_results.db"
DEFAULT_SUMMARY_JSON_NAME = "merged_inference_summary.json"


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _prune_work_dir_outputs(work_dir, keep_filenames=None):
    keep_filenames = set(keep_filenames or [])
    for name in os.listdir(work_dir):
        if name in keep_filenames:
            continue
        path = os.path.join(work_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def get_dm_info_from_npy(
    ase_db_path,
    npy_folder_path,
    convert_smiles_flag=False,
    convention="def2svp",
    mol_charge=0,
    pred_dm_filename="predicted.npy",
    transform_dm_flag=True,
    get_esp_sta_flag=True,
    get_dm_cube_flag=False,
    dm_cube_src="pyscf",
    keep_xyz_file=True,
    max_cube_save: int = 5,
    max_items: int = 300,
    dm_grid: int = 40,
):
    print("Start DM postprocess")
    basis, back_convention = resolve_basis_and_convention(convention)

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()
    all_dm_info = []
    start_time = time.time()

    with connect(ase_db_path) as db:
        for idx, row in tqdm(enumerate(db.select())):
            if idx == max_items:
                break

            work_dir = os.path.join(npy_folder_path, f"{idx}")
            if not os.path.exists(work_dir):
                continue
            os.chdir(work_dir)

            try:
                atom_nums = row.numbers
                an_atoms = row.toatoms()
                if convert_smiles_flag:
                    smiles = atom_2_smile(an_atoms)

                current_mol_charge = get_row_charge(row, mol_charge)
                pred_dm = load_npy_safe(pred_dm_filename)
                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)

                mol, _, _ = build_pyscf_molecule(an_atoms, basis=basis, charge=current_mol_charge, atom_nums=atom_nums)

                multiwfn_gen_dm_flag = False
                if get_dm_cube_flag and idx < max_cube_save:
                    if dm_cube_src == "pyscf":
                        tools.cubegen.density(
                            mol,
                            "pred_electron_density.cube",
                            pred_dm,
                            nx=dm_grid,
                            ny=dm_grid,
                            nz=dm_grid,
                        )
                        tools.cubegen.mep(
                            mol,
                            "pred_molecular_electrostatic_potential.cube",
                            pred_dm,
                            nx=dm_grid,
                            ny=dm_grid,
                            nz=dm_grid,
                        )
                    else:
                        multiwfn_gen_dm_flag = True

                pred_esp_max, pred_esp_min = 0, 0
                if get_esp_sta_flag:
                    pred_esp_max, pred_esp_min = calculate_esp_from_dm(
                        mol, pred_dm, "pred", multiwfn_gen_dm_flag
                    )

                mol_dip = dip_moment(mol, pred_dm, unit="DEBYE")
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                to_sig4 = lambda x: float(f"{x:.4g}")

                dm_info = {
                    "Index": idx,
                    "SMILES": smiles if convert_smiles_flag else "",
                    "Charge": current_mol_charge,
                    "Dipole-X-Debye": to_sig4(mol_dip[0]),
                    "Dipole-Y-Debye": to_sig4(mol_dip[1]),
                    "Dipole-Z-Debye": to_sig4(mol_dip[2]),
                    "Dipole-Moment-magnitude-Debye": to_sig4(dip_magnitude),
                    "ESP-Max-eV": to_sig4(pred_esp_max),
                    "ESP-Min-eV": to_sig4(pred_esp_min),
                }
                if convert_smiles_flag:
                    dm_info["SMILES"] = smiles

                all_dm_info.append(dm_info)

                if keep_xyz_file:
                    write("atomic_structure.xyz", an_atoms)

                with open("dm_info.json", "w") as f_obj:
                    json.dump(dm_info, fp=f_obj, indent=4)

            except Exception as exc:
                print(f"Failed at idx {idx}: {exc}")
                traceback.print_exc()

    end_time = time.time()
    os.chdir(cwd_)

    if all_dm_info:
        df = pd.DataFrame(all_dm_info)
        output_csv_path = os.path.join(cwd_, "dm_summary.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved DM summary to {output_csv_path}")

    second_per_item = (end_time - start_time) / max(1, len(all_dm_info))
    print(f"DM Post-process Time (s/item): {second_per_item}")


def get_ham_info_from_npy(
    ase_db_path,
    npy_folder_path,
    convert_smiles_flag=False,
    convention="def2svp",
    mol_charge=0,
    pred_ham_filename="predicted.npy",
    max_items: int = 300,
    max_cube_save: int = 5,
    cube_grid: int = 40,
):
    print("Start Hamiltonian postprocess")
    basis, back_convention = resolve_basis_and_convention(convention)

    npy_folder_path = os.path.abspath(npy_folder_path)
    cwd_ = os.getcwd()
    all_ham_info = []
    start_time = time.time()

    with connect(ase_db_path) as db:
        for idx, row in tqdm(enumerate(db.select())):
            if idx >= max_items:
                break

            work_dir = os.path.join(npy_folder_path, f"{idx}")
            if not os.path.exists(work_dir):
                continue
            os.chdir(work_dir)

            try:
                atom_nums = row.numbers
                an_atoms = row.toatoms()
                if convert_smiles_flag:
                    smiles = atom_2_smile(an_atoms)

                current_mol_charge = get_row_charge(row, mol_charge)
                pred_ham = load_npy_safe(pred_ham_filename)
                pred_ham_pyscf = matrix_transform(pred_ham, atom_nums, convention=back_convention)

                mol, _, _ = build_pyscf_molecule(an_atoms, basis=basis, charge=current_mol_charge, atom_nums=atom_nums)
                overlap = mol.intor("int1e_ovlp")

                props = get_electronic_properties(mol, ham=pred_ham_pyscf, overlap=overlap, dm=None)
                homo_ev = props["HOMO"] * Hartree
                lumo_ev = props["LUMO"] * Hartree
                gap_ev = props["GAP"] * Hartree

                if idx < max_cube_save:
                    tools.cubegen.orbital(
                        mol,
                        "homo.cube",
                        props["HOMO_coefficients"],
                        nx=cube_grid,
                        ny=cube_grid,
                        nz=cube_grid,
                    )
                    tools.cubegen.orbital(
                        mol,
                        "lumo.cube",
                        props["LUMO_coefficients"],
                        nx=cube_grid,
                        ny=cube_grid,
                        nz=cube_grid,
                    )

                to_sig4 = lambda x: float(f"{x:.4g}")
                ham_info = {
                    "Index": idx,
                    "Charge": current_mol_charge,
                    "HOMO_eV": to_sig4(homo_ev),
                    "LUMO_eV": to_sig4(lumo_ev),
                    "GAP_eV": to_sig4(gap_ev),
                }
                if convert_smiles_flag:
                    ham_info["SMILES"] = smiles

                all_ham_info.append(ham_info)

                with open("ham_info.json", "w") as f_obj:
                    json.dump(ham_info, fp=f_obj, indent=4)

            except Exception as exc:
                print(f"Hamiltonian processing failed at idx {idx}: {exc}")
                traceback.print_exc()

    end_time = time.time()
    os.chdir(cwd_)

    if all_ham_info:
        df = pd.DataFrame(all_ham_info)
        output_csv_path = os.path.join(cwd_, "ham_summary.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved Hamiltonian summary to {output_csv_path}")

    second_per_item = (end_time - start_time) / max(1, len(all_ham_info))
    print(f"Hamiltonian Post-process Time (s/item): {second_per_item}")


def _calc_energy_and_properties(mol, dm, overlap, mf, pcm_eps=None):
    h1e = mf.get_hcore()
    veff = mf.get_veff(mol, dm)
    fock = mf.get_fock(h1e=h1e, vhf=veff, dm=dm)
    tot_energy_hartree = mf.energy_tot(dm=dm, h1e=h1e, vhf=veff)

    pcm_energy_hartree = 0.0
    if hasattr(mf, "with_solvent") and hasattr(mf.with_solvent, "e"):
        pcm_energy_hartree = getattr(mf.with_solvent, "e")

    elec_info = get_electronic_properties(
        mol, ham=fock, dm=dm, overlap=overlap, pcm_eps=pcm_eps, mf=mf
    )
    return elec_info, tot_energy_hartree, pcm_energy_hartree


def save_inference_summary_npz(summary_data_list, results_folder_path, summary_filename):
    if not summary_filename:
        return None
    if not summary_data_list:
        return None

    all_keys = set().union(*(item.keys() for item in summary_data_list))
    npz_dict = {}
    for key in all_keys:
        values = []
        for item in summary_data_list:
            value = item.get(key, np.nan)
            if isinstance(value, (list, np.ndarray)):
                values.append(value)
            elif value is None:
                values.append(np.nan)
            else:
                values.append(value)
        try:
            npz_dict[key] = np.array(values)
        except Exception:
            npz_dict[key] = np.array(values, dtype=object)

    save_path = os.path.join(results_folder_path, summary_filename)
    np.savez(save_path, **npz_dict)
    print(f"[dm_infer_entry] Summary saved to {summary_filename}")
    return save_path


def _as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_optional_output_path(results_folder_path, path_or_auto, default_name):
    if path_or_auto in (None, False):
        return None
    if path_or_auto == "auto":
        return os.path.join(results_folder_path, default_name)
    return os.path.abspath(path_or_auto)


def write_inference_summary_json(summary_data_list, results_folder_path, summary_json_name):
    summary_json_path = _resolve_optional_output_path(
        results_folder_path=results_folder_path,
        path_or_auto=summary_json_name,
        default_name=DEFAULT_SUMMARY_JSON_NAME,
    )
    if not summary_json_path or not summary_data_list:
        return None

    payload = [{key: _as_jsonable(value) for key, value in item.items()} for item in summary_data_list]
    with open(summary_json_path, "w", encoding="utf-8") as f_obj:
        json.dump(payload, f_obj, indent=2)
    return summary_json_path


def _summary_item_to_db_payload(item):
    payload = {"dm_infer_success": True}
    for key, value in item.items():
        if key == "idx":
            payload["dm_infer_idx"] = int(value)
            continue
        if key == "source_idx":
            payload["dm_infer_source_idx"] = int(value)
            continue
        if key == "worker_id":
            continue
        payload[key] = _as_jsonable(value)
    return payload


def write_dm_inference_ase_db(src_ase_path, summary_data_list, dump_ase_db_path):
    dump_ase_db_path = os.path.abspath(dump_ase_db_path)
    for path in (dump_ase_db_path, f"{dump_ase_db_path}-shm", f"{dump_ase_db_path}-wal"):
        if os.path.exists(path):
            os.remove(path)

    summary_by_idx = {}
    for item in summary_data_list:
        raw_idx = item.get("source_idx", item.get("idx"))
        if raw_idx is None:
            continue
        summary_by_idx[int(raw_idx)] = _summary_item_to_db_payload(item)

    updated = 0
    with connect(src_ase_path) as src_db, connect(dump_ase_db_path) as dump_db:
        for idx, row in enumerate(src_db.select()):
            atoms = row.toatoms()
            data = dict(getattr(row, "data", None) or {})
            kvp = dict(getattr(row, "key_value_pairs", None) or {})

            update_payload = summary_by_idx.get(idx)
            if update_payload is not None:
                data.update(update_payload)
                updated += 1

            dump_db.write(atoms, key_value_pairs=kvp, data=data)

    print(f"[dm_infer_entry] Updated ASE DB saved to {dump_ase_db_path} ({updated} rows updated)")
    return dump_ase_db_path


def _get_row_lookup_keys(row, idx):
    data = getattr(row, "data", None) or {}
    keys = []
    for candidate in (data.get("source_row_id"), getattr(row, "id", None), data.get("source_idx"), idx):
        if candidate is None:
            continue
        try:
            candidate = int(candidate)
        except (TypeError, ValueError):
            continue
        if candidate not in keys:
            keys.append(candidate)
    return keys


def _load_matrix_from_work_dir(work_dir, matrix_filename):
    load_path = matrix_filename if os.path.isabs(matrix_filename) else os.path.join(work_dir, matrix_filename)
    return np.asarray(load_npy_safe(load_path))


def _load_matrix_from_lmdb(infer_lmdb_path, row, idx, matrix_field):
    record = None
    used_key = None
    for key in _get_row_lookup_keys(row, idx):
        record = get_pickle_record_any(infer_lmdb_path, key, default=None)
        if record is not None:
            used_key = key
            break
    if record is None:
        raise KeyError(f"LMDB record not found for row id={getattr(row, 'id', None)} idx={idx}")
    if matrix_field not in record:
        raise KeyError(
            f"Matrix field '{matrix_field}' missing in LMDB record for key={used_key}"
        )
    return np.asarray(record[matrix_field])


def _run_dm_infer_entry(
    abs_ase_path,
    results_folder_path,
    matrix_loader,
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    save_cube_info=True,
    n_save_cube_items=5,
    cube_grid=75,
    temp_cube_file="infer_cube_data.pkl",
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    gen_esp_cube_flag=False,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    require_existing_work_dir=False,
    updated_ase_db_path="auto",
    keep_aux_files=True,
):
    basis, back_convention = resolve_basis_and_convention(convention)
    results_folder_path = os.path.abspath(results_folder_path)
    os.makedirs(results_folder_path, exist_ok=True)

    start_time = time.time()
    summary_data_list = []
    temp_cube_data = []
    processed = 0
    failed = 0

    with connect(abs_ase_path) as db:
        total_rows = db.count() if max_items is None else max_items
        for idx, row in tqdm(enumerate(db.select()), total=total_rows):
            if max_items is not None and idx >= max_items:
                break

            work_dir = os.path.join(results_folder_path, f"{idx}")
            if require_existing_work_dir and not os.path.exists(work_dir):
                continue
            os.makedirs(work_dir, exist_ok=True)

            cwd_ = os.getcwd()
            try:
                os.chdir(work_dir)
                processed += 1

                atom_nums = row.numbers
                an_atoms = row.toatoms()
                current_mol_charge = get_row_charge(row, mol_charge)
                current_eps = get_row_dielectric_constant(row, 0)

                print(f"[{idx}] charge = {current_mol_charge}, eps = {current_eps}")

                mol, total_electrons, mol_spin = build_pyscf_molecule(
                    an_atoms, basis=basis, charge=current_mol_charge, atom_nums=atom_nums
                )
                overlap = mol.intor("int1e_ovlp")

                pred_dm = matrix_loader(row=row, idx=idx, work_dir=work_dir)
                if transform_dm_flag:
                    pred_dm = matrix_transform(pred_dm, atom_nums, convention=back_convention)

                mf_pcm = dft.RKS(mol)
                mf_pcm.xc = "b3lyp"
                if current_eps is not None and float(current_eps) > 1.0:
                    mf_pcm = mf_pcm.PCM()
                    mf_pcm.with_solvent.eps = float(current_eps)
                    mf_pcm.with_solvent.method = "IEF-PCM"
                    mf_pcm.with_solvent.radii_table = 1.1 * build_uff_radii_table()
                    mf_pcm.with_solvent.lebedev_order = 31

                if unified_pcm_flag:
                    mf_gas = mf_pcm
                else:
                    mf_gas = dft.RKS(mol)
                    mf_gas.xc = "b3lyp"

                props = {"idx": idx, "charge": int(current_mol_charge), "spin": int(mol_spin)}
                dip_vec = np.asarray(get_dipole_info(mol, pred_dm), dtype=float)
                props["dipole_vector"] = dip_vec.tolist()
                props["dipole_magnitude"] = float(np.linalg.norm(dip_vec))

                zsum = int(an_atoms.get_atomic_numbers().sum())
                ne_pred = get_electron_number_from_dm(pred_dm, overlap)
                props["Ne_actual"] = float(total_electrons)
                props["Ne_pred"] = float(ne_pred)
                props["Ne_error"] = float(abs(ne_pred - total_electrons))
                props["charge_from_dm_infer"] = float(zsum - ne_pred)

                electronic_info_gas = None
                if calc_electronic_flag:
                    elec_info_pcm, e_tot_pcm, e_pcm_int = _calc_energy_and_properties(
                        mol, pred_dm, overlap, mf_pcm, float(current_eps) if current_eps else None
                    )

                    props["total_energy_Hartree"] = float(e_tot_pcm)
                    props["total_energy_eV"] = float(e_tot_pcm * Hartree)
                    if current_eps is not None and float(current_eps) > 1.0:
                        props["pcm_interaction_energy_Hartree"] = float(e_pcm_int)
                        props["pcm_interaction_energy_eV"] = float(e_pcm_int * Hartree)

                    props["HOMO"] = float(elec_info_pcm["HOMO"] * Hartree)
                    props["LUMO"] = float(elec_info_pcm["LUMO"] * Hartree)
                    props["GAP"] = float(elec_info_pcm["GAP"] * Hartree)

                    if unified_pcm_flag:
                        electronic_info_gas = elec_info_pcm
                    else:
                        electronic_info_gas, _, _ = _calc_energy_and_properties(
                            mol, pred_dm, overlap, mf_gas, pcm_eps=None
                        )

                    if save_cube_info and idx < n_save_cube_items:
                        tools.cubegen.orbital(
                            mol,
                            "homo.cube",
                            electronic_info_gas["HOMO_coefficients"],
                            nx=cube_grid,
                            ny=cube_grid,
                            nz=cube_grid,
                        )
                        tools.cubegen.orbital(
                            mol,
                            "lumo.cube",
                            electronic_info_gas["LUMO_coefficients"],
                            nx=cube_grid,
                            ny=cube_grid,
                            nz=cube_grid,
                        )
                        tools.cubegen.density(
                            mol,
                            "pred_density.cube",
                            pred_dm,
                            nx=cube_grid,
                            ny=cube_grid,
                            nz=cube_grid,
                        )

                if calc_fragment_hosting_flag and electronic_info_gas is not None:
                    props.update(
                        infer_orbital_fragment_hosting(
                            row=row,
                            atoms=an_atoms,
                            mol=mol,
                            overlap=electronic_info_gas.get("overlap", overlap),
                            homo_coefficients=electronic_info_gas.get("HOMO_coefficients"),
                            lumo_coefficients=electronic_info_gas.get("LUMO_coefficients"),
                            connectivity_mult=fragment_connectivity_mult,
                        )
                    )

                if calc_esp_flag:
                    kwargs = {
                        "mol": mol,
                        "dm": pred_dm,
                        "prefix": "infer",
                        "gen_dm_flag": gen_esp_cube_flag,
                        "mf": mf_gas,
                        "overlap": overlap,
                    }
                    if electronic_info_gas is not None:
                        kwargs.update(
                            {
                                "fock": electronic_info_gas.get("hamiltonian", None),
                                "mo_energy": electronic_info_gas.get("mo_energy", None),
                                "mo_coeff": electronic_info_gas.get("mo_coeff", None),
                                "mo_occ": electronic_info_gas.get("mo_occ", None),
                            }
                        )

                    esp_max, esp_min, phi = calculate_properties_from_dm(**kwargs)
                    props["ESP_max_eV"] = _optional_float(esp_max)
                    props["ESP_min_eV"] = _optional_float(esp_min)
                    props["Deformation_phi"] = _optional_float(phi)

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
                        {"idx": idx, "mol_info": mol_info, "outputs": electronic_info_gas, "tgt_info": None}
                    )

            except Exception as exc:
                failed += 1
                traceback.print_exc()
                print(f"[dm_infer_entry] idx {idx} failed: {repr(exc)}")
            finally:
                if not keep_aux_files:
                    _prune_work_dir_outputs(work_dir, keep_filenames={"dm_inference_result.json"})
                os.chdir(cwd_)

    if save_cube_info and temp_cube_file and temp_cube_data:
        save_path = os.path.join(results_folder_path, temp_cube_file)
        try:
            with open(save_path, "wb") as f_obj:
                pickle.dump(temp_cube_data, f_obj)
        except Exception:
            pass

    if summary_data_list:
        write_inference_summary_json(
            summary_data_list=summary_data_list,
            results_folder_path=results_folder_path,
            summary_json_name=summary_json_name,
        )
        save_inference_summary_npz(
            summary_data_list=summary_data_list,
            results_folder_path=results_folder_path,
            summary_filename=summary_filename,
        )
        resolved_updated_ase_db_path = _resolve_optional_output_path(
            results_folder_path=results_folder_path,
            path_or_auto=updated_ase_db_path,
            default_name=DEFAULT_UPDATED_ASE_DB_NAME,
        )
        if resolved_updated_ase_db_path:
            write_dm_inference_ase_db(
                src_ase_path=abs_ase_path,
                summary_data_list=summary_data_list,
                dump_ase_db_path=resolved_updated_ase_db_path,
            )

    total_time = time.time() - start_time
    print(f"[dm_infer_entry] Finished. Processed: {processed}, Failed: {failed}")
    print(f"Total Time: {total_time:.2f}s, Avg: {total_time / max(1, processed):.4f}s/item")
    return summary_data_list


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
    cube_grid=75,
    temp_cube_file="infer_cube_data.pkl",
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    gen_esp_cube_flag=False,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    updated_ase_db_path="auto",
    keep_aux_files=True,
):
    return _run_dm_infer_entry(
        abs_ase_path=abs_ase_path,
        results_folder_path=results_folder_path,
        matrix_loader=lambda row, idx, work_dir: _load_matrix_from_work_dir(work_dir, dm_filename),
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        save_cube_info=save_cube_info,
        n_save_cube_items=n_save_cube_items,
        cube_grid=cube_grid,
        temp_cube_file=temp_cube_file,
        summary_filename=summary_filename,
        summary_json_name=summary_json_name,
        gen_esp_cube_flag=gen_esp_cube_flag,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        require_existing_work_dir=True,
        updated_ase_db_path=updated_ase_db_path,
        keep_aux_files=keep_aux_files,
    )


def dm_infer_entry_from_lmdb(
    abs_ase_path,
    infer_lmdb_path,
    results_folder_path,
    matrix_field="hamiltonian",
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    save_cube_info=True,
    n_save_cube_items=5,
    cube_grid=75,
    temp_cube_file="infer_cube_data.pkl",
    summary_filename="inference_summary.npz",
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    gen_esp_cube_flag=False,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    updated_ase_db_path="auto",
    keep_aux_files=True,
):
    infer_lmdb_path = os.path.abspath(infer_lmdb_path)
    return _run_dm_infer_entry(
        abs_ase_path=abs_ase_path,
        results_folder_path=results_folder_path,
        matrix_loader=lambda row, idx, work_dir: _load_matrix_from_lmdb(
            infer_lmdb_path=infer_lmdb_path,
            row=row,
            idx=idx,
            matrix_field=matrix_field,
        ),
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        save_cube_info=save_cube_info,
        n_save_cube_items=n_save_cube_items,
        cube_grid=cube_grid,
        temp_cube_file=temp_cube_file,
        summary_filename=summary_filename,
        summary_json_name=summary_json_name,
        gen_esp_cube_flag=gen_esp_cube_flag,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        require_existing_work_dir=False,
        updated_ase_db_path=updated_ase_db_path,
        keep_aux_files=keep_aux_files,
    )


def dm_infer_light_entry(
    abs_ase_path,
    results_folder_path,
    dm_filename="predicted.npy",
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return dm_infer_entry(
        abs_ase_path=abs_ase_path,
        results_folder_path=results_folder_path,
        dm_filename=dm_filename,
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        save_cube_info=False,
        n_save_cube_items=0,
        temp_cube_file=None,
        summary_filename=None,
        summary_json_name=summary_json_name,
        gen_esp_cube_flag=False,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        updated_ase_db_path=updated_ase_db_path,
        keep_aux_files=False,
    )


def dm_infer_light_entry_from_lmdb(
    abs_ase_path,
    infer_lmdb_path,
    results_folder_path,
    matrix_field="hamiltonian",
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return dm_infer_entry_from_lmdb(
        abs_ase_path=abs_ase_path,
        infer_lmdb_path=infer_lmdb_path,
        results_folder_path=results_folder_path,
        matrix_field=matrix_field,
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        save_cube_info=False,
        n_save_cube_items=0,
        temp_cube_file=None,
        summary_filename=None,
        summary_json_name=summary_json_name,
        gen_esp_cube_flag=False,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        updated_ase_db_path=updated_ase_db_path,
        keep_aux_files=False,
    )


def dm_infer_lightning_entry(
    abs_ase_path,
    results_folder_path,
    dm_filename="predicted.npy",
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return dm_infer_light_entry(
        abs_ase_path=abs_ase_path,
        results_folder_path=results_folder_path,
        dm_filename=dm_filename,
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        summary_json_name=summary_json_name,
        updated_ase_db_path=updated_ase_db_path,
    )


def dm_infer_lightning_entry_from_lmdb(
    abs_ase_path,
    infer_lmdb_path,
    results_folder_path,
    matrix_field="hamiltonian",
    convention="def2svp",
    mol_charge=0,
    transform_dm_flag=True,
    calc_esp_flag=True,
    calc_electronic_flag=True,
    max_items=None,
    unified_pcm_flag=True,
    calc_fragment_hosting_flag=False,
    fragment_connectivity_mult=1.1,
    summary_json_name=DEFAULT_SUMMARY_JSON_NAME,
    updated_ase_db_path="auto",
):
    return dm_infer_light_entry_from_lmdb(
        abs_ase_path=abs_ase_path,
        infer_lmdb_path=infer_lmdb_path,
        results_folder_path=results_folder_path,
        matrix_field=matrix_field,
        convention=convention,
        mol_charge=mol_charge,
        transform_dm_flag=transform_dm_flag,
        calc_esp_flag=calc_esp_flag,
        calc_electronic_flag=calc_electronic_flag,
        max_items=max_items,
        unified_pcm_flag=unified_pcm_flag,
        calc_fragment_hosting_flag=calc_fragment_hosting_flag,
        fragment_connectivity_mult=fragment_connectivity_mult,
        summary_json_name=summary_json_name,
        updated_ase_db_path=updated_ase_db_path,
    )
