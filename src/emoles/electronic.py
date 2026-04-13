import json

import numpy as np
from pyscf.data import radii
from pyscf.scf.hf import make_rdm1

from emoles.pyscf import get_dipole_info
from emoles.utils import get_mo_occ, matrix_transform


UFF_RADII_ANG = {
    1: 1.4430,
    2: 1.1810,
    3: 1.2255,
    4: 1.3725,
    5: 1.8150,
    6: 1.9255,
    7: 1.8300,
    8: 1.7500,
    9: 1.6820,
    10: 1.6215,
    11: 1.4915,
    12: 1.5105,
    13: 2.2495,
    14: 2.1475,
    15: 2.0735,
    16: 2.0175,
    17: 2.0450,
    18: 1.9340,
    19: 1.9060,
    20: 1.6995,
    21: 1.6475,
    22: 1.5875,
    23: 1.5720,
    24: 1.5115,
    25: 1.4805,
    26: 1.4560,
    27: 1.4360,
    28: 1.4170,
    29: 1.7475,
    30: 1.3815,
    31: 2.1915,
    32: 2.1400,
    33: 2.1150,
    34: 2.1025,
    35: 2.1650,
    36: 2.0200,
    37: 2.2585,
    38: 2.0515,
    39: 1.8245,
    40: 1.6155,
    41: 1.5720,
    42: 1.5260,
    43: 1.4990,
    44: 1.4815,
    45: 1.4645,
    46: 1.4495,
    47: 1.5740,
    48: 1.4240,
    49: 2.2315,
    50: 2.1960,
    51: 2.2100,
    52: 2.2350,
    53: 2.3600,
    54: 2.1815,
}

BOHR = radii.BOHR


def build_uff_radii_table():
    """UFF radii table in Bohr."""
    table = np.zeros(118)
    for atomic_number, radius_ang in UFF_RADII_ANG.items():
        table[atomic_number] = radius_ang / BOHR
    return table


def calculate_dm_dipole_mae(pred_dm, target_dm, mol):
    error_dict = {}
    diff_matrix = np.abs(np.array(pred_dm - target_dm))
    error_dict["density_matrix"] = np.mean(diff_matrix)
    dip_pred = get_dipole_info(mol, pred_dm)
    dip_target = get_dipole_info(mol, target_dm)
    error_dict["dipole"] = np.abs(np.array(dip_pred - dip_target))
    return error_dict


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian):
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)
    eps = 1e-8 * np.ones_like(eigvals)
    eigvals = np.where(eigvals > 1e-8, eigvals, eps)
    frac_overlap = eigvecs / np.sqrt(eigvals[:, np.newaxis])

    fs = np.matmul(
        np.matmul(np.transpose(frac_overlap, (0, 2, 1)), full_hamiltonian),
        frac_overlap,
    )
    orbital_energies, orbital_coefficients = np.linalg.eigh(fs)
    orbital_coefficients = frac_overlap @ orbital_coefficients
    return orbital_energies[0], orbital_coefficients[0]


def get_electron_number_from_dm(dm, overlap):
    p_matrix = dm
    s_matrix = overlap

    if p_matrix.ndim == 3:
        p_matrix = p_matrix.sum(axis=0)

    if s_matrix.ndim == 3:
        s_matrix = s_matrix[0]

    return float(np.einsum("ij,ji->", p_matrix, s_matrix))


def prepare_np(
    overlap_matrix,
    full_hamiltonian,
    atom_symbols,
    transform_ham_flag=False,
    transform_overlap_flag=False,
    convention="def2svp",
):
    if convention == "6311gdp":
        back_convention = "back_2_thu_pyscf"
    else:
        back_convention = "back2pyscf"

    overlap_matrix = np.expand_dims(overlap_matrix, axis=0)
    full_hamiltonian = np.expand_dims(full_hamiltonian, axis=0)
    if transform_ham_flag:
        full_hamiltonian = matrix_transform(
            full_hamiltonian, atom_symbols, convention=back_convention
        )
    if transform_overlap_flag:
        overlap_matrix = matrix_transform(
            overlap_matrix, atom_symbols, convention=back_convention
        )
    return full_hamiltonian, overlap_matrix


def get_electronic_properties(
    mol,
    ham=None,
    overlap=None,
    dm=None,
    shifted_ham=None,
    pcm_eps=1.0,
    mf=None,
):
    from pyscf import dft

    if ham is None:
        if dm is None:
            raise ValueError("Must provide either Hamiltonian or Density Matrix")

        if mf is None:
            mf = dft.RKS(mol)
            mf.xc = "b3lyp"
            if pcm_eps > 1.0:
                mf = mf.PCM()
                mf.with_solvent.eps = pcm_eps
                mf.with_solvent.method = "IEF-PCM"
                uff_radii = build_uff_radii_table()
                mf.with_solvent.radii_table = 1.1 * uff_radii
                mf.with_solvent.lebedev_order = 31

        ham = mf.get_fock(dm=dm)
        if overlap is None:
            overlap = mf.get_ovlp()

    if overlap is None:
        overlap = mol.intor("int1e_ovlp")

    ham_2d = ham[0] if getattr(ham, "ndim", 0) == 3 else ham
    ov_2d = overlap[0] if getattr(overlap, "ndim", 0) == 3 else overlap

    ham_in_3d = ham_2d[None, ...]
    ov_in_3d = ov_2d[None, ...]

    energies, coeffs = cal_orbital_and_energies(
        overlap_matrix=ov_in_3d, full_hamiltonian=ham_in_3d
    )

    n_electrons = mol.tot_electrons()
    homo_idx = int(n_electrons / 2) - 1
    mo_occ = get_mo_occ(full_len=len(energies), occ_len=homo_idx + 1)

    if shifted_ham is None:
        shifted_ham_2d = ham_2d
    else:
        shifted_ham_2d = shifted_ham[0] if getattr(shifted_ham, "ndim", 0) == 3 else shifted_ham

    return {
        "HOMO": energies[homo_idx],
        "LUMO": energies[homo_idx + 1],
        "GAP": energies[homo_idx + 1] - energies[homo_idx],
        "hamiltonian": ham_in_3d,
        "overlap": ov_in_3d,
        "shifted_ham": shifted_ham_2d,
        "density_matrix": dm if dm is not None else make_rdm1(mo_coeff=coeffs, mo_occ=mo_occ),
        "mo_occ": mo_occ,
        "mo_energy": energies,
        "mo_coeff": coeffs,
        "orbital_coefficients": coeffs[:, : homo_idx + 1],
        "HOMO_coefficients": coeffs[:, homo_idx],
        "LUMO_coefficients": coeffs[:, homo_idx + 1],
        "occupied_orbital_energy": energies[: homo_idx + 1],
    }


def calculate_esp_from_dm(mol, dm, prefix, gen_dm_flag=False):
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft
    from emoles.multiwfn import ESPCalculator

    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    fock = mf.get_fock(dm=dm)
    overlap = mf.get_ovlp()

    orbital_energies, orbital_coefficients = mf.eig(fock, overlap)
    mf.mo_energy = orbital_energies
    mf.mo_coeff = orbital_coefficients
    mf.dm = dm

    fch_filename = f"{prefix}.fch"
    fchk(mf, fch_filename, density=True)

    esp_calculator = ESPCalculator(fch_filename)
    esp_results = esp_calculator.get_ESP_value()
    if gen_dm_flag:
        esp_calculator.get_acc_grid_data()

    with open(f"{prefix}_esp_info.json", "w") as f_obj:
        json.dump(esp_results, fp=f_obj, indent=4)

    esp_max = esp_results.get("ESP_max_eV", 0)
    esp_min = esp_results.get("ESP_min_eV", 0)
    return esp_max, esp_min


def calculate_properties_from_dm(
    mol,
    dm,
    prefix,
    gen_dm_flag=False,
    mf=None,
    fock=None,
    overlap=None,
    mo_energy=None,
    mo_coeff=None,
    mo_occ=None,
    xc="b3lyp",
    pcm_eps=1.0,
):
    from mokit.lib.py2fch_direct import fchk
    from pyscf import dft
    from emoles.multiwfn import ELFDeformationCalculator, ESPCalculator

    if mf is None:
        mf = dft.RKS(mol)
        mf.xc = xc
        if pcm_eps > 1.0:
            mf = mf.PCM()
            mf.with_solvent.eps = pcm_eps
            mf.with_solvent.method = "IEF-PCM"
            uff_radii = build_uff_radii_table()
            mf.with_solvent.radii_table = 1.1 * uff_radii
            mf.with_solvent.lebedev_order = 31

    if overlap is None:
        try:
            overlap = mf.get_ovlp()
        except Exception:
            overlap = mol.intor("int1e_ovlp")

    ov_2d = overlap[0] if getattr(overlap, "ndim", 0) == 3 else overlap

    if (mo_energy is None) or (mo_coeff is None):
        if fock is None:
            fock = mf.get_fock(dm=dm)
        fock_2d = fock[0] if getattr(fock, "ndim", 0) == 3 else fock
        mo_energy, mo_coeff = mf.eig(fock_2d, ov_2d)

    if mo_occ is None:
        n_electrons = mol.tot_electrons()
        homo_idx = int(n_electrons / 2) - 1
        mo_occ = get_mo_occ(full_len=len(mo_energy), occ_len=homo_idx + 1)

    mf.mo_energy = np.array(mo_energy)
    mf.mo_coeff = np.array(mo_coeff)
    mf.mo_occ = np.array(mo_occ)
    mf.dm = dm

    fch_filename = f"{prefix}.fch"
    fchk(mf, fch_filename, density=True)

    esp_calculator = ESPCalculator(fch_filename)
    esp_results = esp_calculator.get_ESP_value()
    if gen_dm_flag:
        esp_calculator.get_acc_grid_data()

    with open(f"{prefix}_esp_info.json", "w") as f_obj:
        json.dump(esp_results, fp=f_obj, indent=4)

    esp_max = esp_results.get("ESP_max_eV", 0.0)
    esp_min = esp_results.get("ESP_min_eV", 0.0)

    li_phi = None
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords_ang = mol.atom_coords(unit="Ang")
    li_indices_0based = [i for i, symbol in enumerate(symbols) if symbol == "Li"]

    if li_indices_0based:
        target_li_idx = li_indices_0based[0]
        i_1based = target_li_idx + 1
        li_center = coords_ang[target_li_idx]

        elf_calculator = ELFDeformationCalculator(
            fch_filename, isovalue=0.5, diff_list=[0.09], li_cutoff=1.1
        )
        save_id = f"{prefix}_{i_1based}"

        try:
            elf_res = elf_calculator.calculate(
                atom_index_1based=i_1based,
                li_center=li_center,
                li_id=save_id,
                radius=3.0,
                grid_spacing=0.1,
            )
            target_diff = 0.09
            if elf_res and target_diff in elf_res:
                li_phi = elf_res[target_diff]["phi"]
        except Exception as exc:
            print(f"Warning: Failed to calculate phi for {prefix} Li: {exc}")

    return esp_max, esp_min, li_phi
