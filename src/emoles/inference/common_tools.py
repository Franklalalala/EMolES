import os
from collections import defaultdict

import numpy as np

_ELECTRONIC_EXPORTS = {
    "calculate_esp_from_dm",
    "cal_orbital_and_energies",
    "prepare_np",
}

_CHEMISTRY_EXPORTS = {
    "annotate_db_dc_by_similarity",
    "atom_2_mol",
    "atom_2_smile",
    "calculate_with_multiwfn",
    "generate_cube_files",
    "mol_2_atom",
    "smile_2_atom",
    "smile_2_db",
    "smile_to_inchi",
    "smile_to_maccs_fp_arr",
    "tanimoto_similarity",
    "get_overlap_matrix",
    "info_collector",
}


def load_npy_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path)


def resolve_basis_and_convention(convention):
    if convention == "6311gdp":
        return "6-311+g(d,p)", "back_2_thu_pyscf"
    if convention == "back_thu_cluster":
        return "def2svp", "back_thu_cluster"
    return "def2svp", "back2pyscf"


def get_row_charge(row, default=0):
    data = getattr(row, "data", None) or {}
    return data.get("charge", default)


def get_row_dielectric_constant(row, default=0):
    attr_val = getattr(row, "dielectric_constant", None)
    if attr_val is not None:
        return attr_val
    data = getattr(row, "data", None) or {}
    if data.get("dielectric_constant", None) is not None:
        return data["dielectric_constant"]
    return data.get("dielectric_constant_weighted_detail", {}).get(
        "dielectric_constant_weighted", default
    )


def get_row_identifier_payload(row, fallback_idx=None):
    def _maybe_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    data = getattr(row, "data", None) or {}
    key_value_pairs = getattr(row, "key_value_pairs", None) or {}
    merged = {}
    merged.update(dict(key_value_pairs))
    merged.update(dict(data))

    source_idx = merged.get("source_idx", fallback_idx)
    source_row_id = merged.get("source_row_id", getattr(row, "id", None))

    sample_id = None
    for key in (
        "sample_id",
        "test_id",
        "mol_id",
        "molecule_id",
        "structure_id",
        "name",
        "source_row_id",
        "source_idx",
        "id",
        "idx",
    ):
        value = merged.get(key)
        if value is not None:
            sample_id = value
            break

    if sample_id is None:
        sample_id = source_row_id if source_row_id is not None else source_idx

    return {
        "source_idx": _maybe_int(source_idx),
        "source_row_id": _maybe_int(source_row_id),
        "sample_id": sample_id,
    }


def get_row_orbital_labels(row, default=0.0):
    data = getattr(row, "data", None) or {}
    homo = data.get("HOMO_eV", default)
    lumo = data.get("LUMO_eV", default)
    gap = data.get("GAP_eV", lumo - homo)
    return float(homo), float(lumo), float(gap)


def build_pyscf_molecule(an_atoms, basis, charge=0, atom_nums=None):
    import pyscf

    if atom_nums is None:
        atom_nums = an_atoms.numbers

    total_electrons = int(an_atoms.get_atomic_numbers().sum()) - int(charge)
    mol_spin = total_electrons % 2

    mol = pyscf.gto.Mole()
    mol.charge = charge
    mol.spin = mol_spin
    mol.build(
        verbose=0,
        atom=[[atom_nums[i], at.position] for i, at in enumerate(an_atoms)],
        basis=basis,
        unit="ang",
    )
    return mol, total_electrons, mol_spin


def extract_model_params(model):
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return {}, {}

    init_layer = getattr(embedding, "init_layer", None)
    if init_layer is None:
        return {}, {}

    raw_basis = getattr(embedding, "basis", {})
    basis_clean = {}
    orbital_types = ["s", "p", "d", "f"]

    for elem, orb_list in raw_basis.items():
        counts = defaultdict(int)
        for orb in orb_list:
            if orb:
                counts[orb[-1]] += 1

        dense_str = ""
        for orbital_type in orbital_types:
            count = counts[orbital_type]
            if count > 0:
                dense_str += f"{count}{orbital_type}"
        basis_clean[elem] = dense_str

    raw_r_max_dict = getattr(init_layer, "r_max_dict", None)
    raw_r_max_scalar = getattr(init_layer, "r_max", None)

    r_max_clean = {}
    if raw_r_max_dict is not None:
        for elem, tensor_val in raw_r_max_dict.items():
            r_max_clean[elem] = tensor_val.item()
    elif raw_r_max_scalar is not None:
        scalar_val = raw_r_max_scalar.item()
        for elem in basis_clean.keys():
            r_max_clean[elem] = scalar_val

    return basis_clean, r_max_clean


_load_npy_safe = load_npy_safe


def __getattr__(name):
    if name in _ELECTRONIC_EXPORTS:
        from emoles import electronic

        return getattr(electronic, name)
    if name in _CHEMISTRY_EXPORTS:
        from emoles.inference import chemistry

        return getattr(chemistry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "_load_npy_safe",
    "annotate_db_dc_by_similarity",
    "atom_2_mol",
    "atom_2_smile",
    "calculate_esp_from_dm",
    "calculate_with_multiwfn",
    "cal_orbital_and_energies",
    "extract_model_params",
    "build_pyscf_molecule",
    "generate_cube_files",
    "get_overlap_matrix",
    "get_row_charge",
    "get_row_dielectric_constant",
    "get_row_identifier_payload",
    "get_row_orbital_labels",
    "info_collector",
    "load_npy_safe",
    "mol_2_atom",
    "prepare_np",
    "resolve_basis_and_convention",
    "smile_2_atom",
    "smile_2_db",
    "smile_to_inchi",
    "smile_to_maccs_fp_arr",
    "tanimoto_similarity",
]
