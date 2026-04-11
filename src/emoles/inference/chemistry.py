import json
import os

import ase
import numpy as np
import pandas as pd
import rdkit
from ase.atom import Atom
from ase.atoms import Atoms
from ase.db.core import connect
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDetermineBonds import DetermineBonds

from emoles.electronic import calculate_esp_from_dm, cal_orbital_and_energies, prepare_np
from emoles.multiwfn import ESPCalculator
from emoles.utils import matrix_transform


class info_collector:
    def __init__(self):
        self.homo_list = []
        self.lumo_list = []
        self.gap_list = []

    def parse_orbital_energies(self, orbital_energies, homo_index):
        from ase.units import Hartree

        homo = orbital_energies[homo_index] * Hartree
        lumo = orbital_energies[homo_index + 1] * Hartree
        gap = lumo - homo
        self.homo_list.append(homo)
        self.lumo_list.append(lumo)
        self.gap_list.append(gap)

    def dump_to_csv(self, csv_path):
        df = pd.DataFrame({"HOMO": self.homo_list, "LUMO": self.lumo_list, "Gap": self.gap_list})
        df.to_csv(csv_path, index=False)


def get_overlap_matrix(ase_atoms, basis):
    import pyscf

    mol = pyscf.gto.Mole()
    atom_list = [[ase_atoms.numbers[atom_idx], atom.position] for atom_idx, atom in enumerate(ase_atoms)]
    mol.build(verbose=0, atom=atom_list, basis=basis, unit="ang")
    overlap = mol.intor("int1e_ovlp")
    return overlap, mol


def generate_cube_files(
    ase_db_path: str,
    out_path: str,
    n_grid,
    basis="def2svp",
    dm_flag=False,
    keep_xyz_file=True,
    dm_grid: int = 40,
    limit: int = 2,
):
    from pyscf import tools
    from pyscf.scf.hf import dip_moment, make_rdm1

    if basis == "def2svp":
        transform_convention = "back2pyscf"
        overlap_basis = "def2svp"
    elif basis == "6311gdp":
        transform_convention = "back_2_thu_pyscf"
        overlap_basis = "6-311+g(d,p)"
    else:
        raise NotImplementedError

    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, "cube")
    with connect(ase_db_path) as db:
        for idx, row in enumerate(db.select()):
            an_atoms = row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load("predicted_ham.npy")
            hamiltonian, overlap = prepare_np(
                overlap_matrix=overlap,
                full_hamiltonian=predicted_ham,
                atom_symbols=an_atoms.numbers,
                transform_ham_flag=True,
                transform_overlap_flag=False,
                convention="6311gdp" if basis == "6311gdp" else "def2svp",
            )
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(
                overlap_matrix=overlap, full_hamiltonian=hamiltonian
            )
            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(
                orbital_energies=orbital_energies, homo_index=homo_idx
            )
            homo_coefficients = orbital_coefficients[:, homo_idx]
            lumo_coefficients = orbital_coefficients[:, homo_idx + 1]
            tools.cubegen.orbital(mol, "HOMO.cube", homo_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, "LUMO.cube", lumo_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)

            if dm_flag:
                mo_occ = np.zeros(overlap.shape[-1])
                mo_occ[: homo_idx + 1] = 2
                dm = make_rdm1(mo_coeff=orbital_coefficients, mo_occ=mo_occ)
                tools.cubegen.density(mol, "electron_density.cube", dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                tools.cubegen.mep(
                    mol,
                    "molecular_electrostatic_potential.cube",
                    dm,
                    nx=dm_grid,
                    ny=dm_grid,
                    nz=dm_grid,
                )
                mol_dip = dip_moment(mol, dm, unit="DEBYE")
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                dipole_info = {
                    "Dipole_Moment_Vector_DEBYE": mol_dip.tolist(),
                    "Dipole_Moment_Norm_DEBYE": float(dip_magnitude),
                }
                with open("dipole_info.json", "w") as f_obj:
                    json.dump(dipole_info, fp=f_obj)

            if keep_xyz_file:
                write("atomic_structure.xyz", an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, "energy_info.csv")
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)


def calculate_with_multiwfn(
    ase_db_path: str,
    out_path: str,
    n_grid,
    basis="def2svp",
    esp_flag=False,
    keep_xyz_file=True,
    dm_grid: int = 40,
    limit: int = 2,
):
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft

    if basis == "def2svp":
        overlap_basis = "def2svp"
    elif basis == "6311gdp":
        overlap_basis = "6-311+g(d,p)"
    else:
        raise NotImplementedError

    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, "cube")
    with connect(ase_db_path) as db:
        for idx, row in enumerate(db.select()):
            an_atoms = row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load("predicted_ham.npy")
            hamiltonian, overlap = prepare_np(
                overlap_matrix=overlap,
                full_hamiltonian=predicted_ham,
                atom_symbols=an_atoms.numbers,
                transform_ham_flag=True,
                transform_overlap_flag=False,
                convention="6311gdp" if basis == "6311gdp" else "def2svp",
            )
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(
                overlap_matrix=overlap, full_hamiltonian=hamiltonian
            )
            mf = dft.RKS(mol)
            mf.mo_coeff = orbital_coefficients
            mf.mo_energy = orbital_energies
            fchk(mf, "predicted.fch", density=True)

            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(
                orbital_energies=orbital_energies, homo_index=homo_idx
            )

            if esp_flag:
                esp_calculator = ESPCalculator("predicted.fch")
                esp_results, cube_file = esp_calculator.calculate_grid_data()
                with open("esp_info.json", "w") as f_obj:
                    json.dump(esp_results, fp=f_obj)

            if keep_xyz_file:
                write("atomic_structure.xyz", an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, "energy_info.csv")
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)


def mol_2_atom(mol: rdkit.Chem.rdchem.Mol):
    conf = mol.GetConformer()
    an_atoms = Atoms()
    for idx in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(idx)
        atom = mol.GetAtoms()[idx]
        symbol = atom.GetSymbol()
        an_atoms.append(Atom(symbol=symbol, position=(position.x, position.y, position.z)))
    return an_atoms


def atom_2_mol(an_atoms: ase.atoms.Atoms):
    write(filename="temp.xyz", images=an_atoms)
    raw_mol = Chem.MolFromXYZFile("temp.xyz")
    mol = Chem.Mol(raw_mol)
    DetermineBonds(mol, useHueckel=True)
    os.remove("temp.xyz")
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
    intersection = np.logical_and(fp1, fp2).sum()
    union = np.logical_or(fp1, fp2).sum()
    return float(intersection / union) if union > 0 else 0.0


def atom_2_smile(an_atoms: ase.atoms.Atoms):
    mol = atom_2_mol(an_atoms)
    return Chem.MolToSmiles(mol)


def smile_2_atom(smile: str, maxAttempts: int = 1000000):
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=maxAttempts)
    return mol_2_atom(mol)


def annotate_db_dc_by_similarity(q_db_path: str, ref_db_path: str):
    with connect(ref_db_path) as ref_db:
        ref_rows = list(ref_db.select())
        ref_items = []
        for row in ref_rows:
            an_atoms = row.toatoms()
            smile = atom_2_smile(an_atoms)
            inchi = smile_to_inchi(smile)
            fp = smile_to_maccs_fp_arr(smile)
            dc = float(getattr(row, "dielectric_constant", 0.0))
            ref_items.append({"inchi": inchi, "fp": fp, "dc": dc})

    with connect(q_db_path) as q_db:
        for row in q_db.select():
            an_atoms = row.toatoms()
            try:
                q_smile = atom_2_smile(an_atoms)
                q_inchi = smile_to_inchi(q_smile)
                q_fp = smile_to_maccs_fp_arr(q_smile)
            except Exception:
                continue

            matched_dc = None
            for ref in ref_items:
                if q_inchi == ref["inchi"]:
                    matched_dc = ref["dc"]
                    break
            if matched_dc is None:
                sims = [tanimoto_similarity(q_fp, ref["fp"]) for ref in ref_items]
                matched_dc = ref_items[int(np.argmax(sims))]["dc"]

            data = dict(row.data)
            data["dielectric_constant_weighted"] = matched_dc
            q_db.update(row.id, dielectric_constant=matched_dc, data=data)


def smile_2_db(
    smile_path: str,
    db_path: str,
    fail_smile_path: str,
    maxAttempts: int = 1000000,
):
    with open(smile_path, "r") as f_obj:
        smiles_list = [line.strip() for line in f_obj if line.strip()]

    fail_smiles = []
    if os.path.exists(db_path):
        os.remove(db_path)

    with connect(db_path) as db:
        for smile in smiles_list:
            try:
                an_atoms = smile_2_atom(smile, maxAttempts=maxAttempts)
                db.write(an_atoms, data={"smile": smile})
            except Exception:
                fail_smiles.append(smile)

    with open(fail_smile_path, "w") as f_obj:
        for smile in fail_smiles:
            f_obj.write(smile + "\n")
