import numpy as np


def format_number(num):
    """Format number based on magnitude."""
    abs_num = abs(num)
    if abs_num >= 1:
        return f"{num:.2f}"
    return f"{num:.3g}"


def vec_cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.abs(dot_product / (norm_a * norm_b))


def get_mo_occ(full_len: int, occ_len: int):
    mo_occ = np.zeros(full_len)
    mo_occ[:occ_len] = 2
    return mo_occ
