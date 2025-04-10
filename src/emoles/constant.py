from argparse import Namespace


loss_weights = {
    'hamiltonian': 1.0,
    'diagonal_hamiltonian': 1.0,
    'non_diagonal_hamiltonian': 1.0,
    'orbital_energies': 1.0,
    "orbital_coefficients": 1.0,
    "HOMO_coefficients": 1.0,
    "LUMO_coefficients": 1.0,
    'HOMO': 1.0, 'LUMO': 1.0, 'GAP': 1.0,
}


atom_to_transform_indices = {'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                             'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                             'F': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                             'N': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                             'Li': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                             'H': [0, 1, 2, 3, 4]}


BOHR2ANG = 1.8897259886

convention_dict = {
    'pyscf_def2svp': Namespace(
        atom_to_orbitals_map={1: 'ssp', 3: 'ssspp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'gau_def2svp_2_pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 3: 'ssspp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [0, 1, 2], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        },
    ),
    'pyscf_6311_plus_gdp': Namespace(
        atom_to_orbitals_map={1: 'sssp', 3: 'sssssppppd', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd', 9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),
    'back2pyscf': Namespace(
        atom_to_orbitals_map={1: 'ssp', 3: 'ssspp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd', 9: 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2], 3: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5], 9: [0, 1, 2, 3, 4, 5]
        }
    ),
    'back_2_thu_pyscf': Namespace(
        atom_to_orbitals_map={1: 'sssp', 3: 'sssssppppd', 6: 'sssssppppd', 7: 'sssssppppd', 8: 'sssssppppd', 9: 'sssssppppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2, 3], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
    ),
}

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}