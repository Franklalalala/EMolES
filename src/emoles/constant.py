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
