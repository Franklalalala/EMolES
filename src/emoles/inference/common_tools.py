import os

import json
import numpy as np

from ase.units import Hartree
from emoles.utils import matrix_transform
from ase.db.core import connect
from ase.io import write
from ase.atom import Atom
from ase.atoms import Atoms
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


class info_collector:
    def __init__(self):
        self.homo_list = []
        self.lumo_list = []
        self.gap_list = []
    def parse_orbital_energies(self, orbital_energies, homo_index):
        homo = orbital_energies[homo_index]*Hartree
        lumo = orbital_energies[homo_index+1]*Hartree
        gap = lumo - homo
        self.homo_list.append(homo)
        self.lumo_list.append(lumo)
        self.gap_list.append(gap)
    def dump_to_csv(self, csv_path):
        data = {
            'HOMO': self.homo_list,
            'LUMO': self.lumo_list,
            'Gap': self.gap_list
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    Fs = np.matmul(np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = np.linalg.eigh(Fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def prepare_np(overlap_matrix, full_hamiltonian, atom_numbers, transform_ham_flag=False, transform_overlap_flag=False, transform_convention='back2pyscf'):
    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(full_hamiltonian, atom_numbers, convention=transform_convention)
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(overlap_matrix, atom_numbers, convention=transform_convention)
    return full_hamiltonian, overlap_matrix


def get_overlap_matrix(ase_atoms, basis):
    import pyscf

    mol = pyscf.gto.Mole()
    t = [[ase_atoms.numbers[atom_idx], an_atom.position]
         for atom_idx, an_atom in enumerate(ase_atoms)]
    mol.build(verbose=0, atom=t, basis=basis, unit='ang')
    overlap = mol.intor("int1e_ovlp")
    return overlap, mol


def generate_cube_files(ase_db_path: str, out_path: str, n_grid, basis='def2svp', dm_flag=False, keep_xyz_file: bool=True, dm_grid: int=40, limit: int=2):
    from pyscf.scf.hf import make_rdm1, dip_moment
    from pyscf import tools

    """Generate cube files for HOMO and LUMO orbitals and save them in sub-folders named by idx."""
    if basis == 'def2svp':
        transform_convention = 'back2pyscf'
        overlap_basis = 'def2svp'
    elif basis == '6311gdp':
        transform_convention = 'back_2_thu_pyscf'
        overlap_basis = '6-311+g(d,p)'
    else:
        raise NotImplementedError
    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, 'cube')
    with connect(ase_db_path) as db:
        for idx, a_row in enumerate(db.select()):
            an_atoms = a_row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load('predicted_ham.npy')
            hamiltonian, overlap = prepare_np(overlap_matrix=overlap, full_hamiltonian=predicted_ham, atom_numbers=an_atoms.numbers, transform_ham_flag=True, transform_overlap_flag=False, transform_convention=transform_convention)
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(overlap_matrix=overlap, full_hamiltonian=hamiltonian)
            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(orbital_energies=orbital_energies, homo_index=homo_idx)
            HOMO_coefficients, LUMO_coefficients = orbital_coefficients[:, homo_idx], orbital_coefficients[:, homo_idx+1]
            # tools.cubegen.orbital(mol, 'HOMO_big_margin.cube', HOMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid, margin=9)
            # tools.cubegen.orbital(mol, 'LUMO_big_margin.cube', LUMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid, margin=9)
            tools.cubegen.orbital(mol, 'HOMO.cube', HOMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'LUMO.cube', LUMO_coefficients, nx=n_grid, ny=n_grid, nz=n_grid)

            if dm_flag:
                mo_occ = np.zeros(overlap.shape[-1])
                mo_occ[:homo_idx+1] = 2
                dm = make_rdm1(mo_coeff=orbital_coefficients, mo_occ=mo_occ)
                tools.cubegen.density(mol, 'electron_density.cube', dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                tools.cubegen.mep(mol, 'molecular_electrostatic_potential.cube', dm, nx=dm_grid, ny=dm_grid, nz=dm_grid)
                mol_dip = dip_moment(mol, dm, unit='DEBYE')
                dip_magnitude = np.linalg.norm(np.array(mol_dip))
                dipole_info = {
                    'Dipole_Moment_Vector_DEBYE': mol_dip.tolist(),
                    'Dipole_Moment_Norm_DEBYE': float(dip_magnitude),
                }
                with open('dipole_info.json', 'w') as f:
                    json.dump(dipole_info, fp=f)

            if keep_xyz_file:
                write('atomic_structure.xyz', an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, 'energy_info.csv')
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)


def calculate_with_multiwfn(ase_db_path: str, out_path: str, n_grid, basis='def2svp', esp_flag=False, keep_xyz_file: bool=True, dm_grid: int=40, limit: int=2):
    from pyscf.scf.hf import make_rdm1, dip_moment
    from pyscf import tools, dft
    from emoles.multiwfn import ESPCalculator
    from mokit.lib.py2fch_direct import fchk

    """Generate cube files for HOMO and LUMO orbitals and save them in sub-folders named by idx."""
    if basis == 'def2svp':
        transform_convention = 'back2pyscf'
        overlap_basis = 'def2svp'
    elif basis == '6311gdp':
        transform_convention = 'back_2_thu_pyscf'
        overlap_basis = '6-311+g(d,p)'
    else:
        raise NotImplementedError
    energy_info_collector = info_collector()
    cwd_ = os.getcwd()
    abs_out_path = os.path.abspath(out_path)
    cube_dump_place = os.path.join(abs_out_path, 'cube')
    with connect(ase_db_path) as db:
        for idx, a_row in enumerate(db.select()):
            an_atoms = a_row.toatoms()
            overlap, mol = get_overlap_matrix(ase_atoms=an_atoms, basis=overlap_basis)
            os.chdir(cube_dump_place)
            os.chdir(str(idx))
            predicted_ham = np.load('predicted_ham.npy')
            hamiltonian, overlap = prepare_np(overlap_matrix=overlap, full_hamiltonian=predicted_ham, atom_numbers=an_atoms.numbers, transform_ham_flag=True, transform_overlap_flag=False, transform_convention=transform_convention)
            orbital_energies, orbital_coefficients = cal_orbital_and_energies(overlap_matrix=overlap, full_hamiltonian=hamiltonian)
            mf = dft.RKS(mol)
            mf.mo_coeff = orbital_coefficients
            mf.mo_energy = orbital_energies
            fchk(mf, 'predicted.fch', density=True)

            homo_idx = int(sum(an_atoms.numbers) / 2) - 1
            energy_info_collector.parse_orbital_energies(orbital_energies=orbital_energies, homo_index=homo_idx)
            # HOMO_coefficients, LUMO_coefficients = orbital_coefficients[:, homo_idx], orbital_coefficients[:, homo_idx+1]

            if esp_flag:
                esp_calculator = ESPCalculator("predicted.fch")
                esp_results, cube_file = esp_calculator.calculate_grid_data()
                with open('esp_info.json', 'w') as f:
                    json.dump(esp_results, fp=f)

            if keep_xyz_file:
                write('atomic_structure.xyz', an_atoms)

            if idx == limit - 1:
                break

    csv_path = os.path.join(abs_out_path, 'energy_info.csv')
    energy_info_collector.dump_to_csv(csv_path=csv_path)
    os.chdir(cwd_)


def mol_2_atom(mol: rdkit.Chem.rdchem.Mol):
    conf = mol.GetConformer()
    an_atoms = Atoms()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        atom = mol.GetAtoms()[i]
        a_symbol = atom.GetSymbol()
        an_new_atom = Atom(symbol=a_symbol, position=(position.x, position.y, position.z))
        an_atoms.append(an_new_atom)
    return an_atoms


def smile_2_atom(smile: str, maxAttempts: int=1000000):
    a_mol = Chem.MolFromSmiles(smile)
    a_mol_with_H = Chem.AddHs(a_mol)
    AllChem.EmbedMolecule(a_mol_with_H, useRandomCoords=True, maxAttempts=maxAttempts)
    AllChem.MMFFOptimizeMolecule(a_mol_with_H)
    an_atoms = mol_2_atom(mol=a_mol_with_H)
    return an_atoms


def smile_2_db(smile_path: str, db_path: str, fail_smile_path: str,  maxAttempts: int=1000000):
    print('Each row corresponds to a smile by default.')
    real_count = 0
    fail_count = 0
    with connect(db_path) as db, open(smile_path, 'r') as smi_r:
        for i, a_line in enumerate(smi_r.readlines()):
            a_smile = a_line.strip()
            if a_smile == '':
                print('An empty line is detected.')
                continue
            try:
                an_atoms = smile_2_atom(smile=a_smile, maxAttempts=maxAttempts)
                db.write(atoms=an_atoms, smile=a_smile)
                real_count = real_count + 1
            except Exception as e:
                print(e)
                fail_count = fail_count + 1
                with open(fail_smile_path, 'a') as f_f:
                    f_f.write(a_smile)
                    f_f.write('\n')
    print(f'real: {real_count}')
    print(f'fail: {fail_count}')