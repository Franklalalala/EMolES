import logging

import numpy as np
import torch

from emoles.utils.constants import CUBIC_MAG_NUM_DICT, LM_MAG_NUM_DICT

log = logging.getLogger(__name__)


def lm2cubic_mat(cubic_mag_num, lm_mag_num, device="cpu", dtype=torch.float32):
    assert len(cubic_mag_num) in [1, 3, 5]
    assert len(lm_mag_num) == len(cubic_mag_num)

    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        raise TypeError("Only float32 and float64 are supported.")

    s2_1 = 1.0 / torch.sqrt(torch.tensor(2.0))
    matrix = torch.zeros(
        [len(cubic_mag_num), len(cubic_mag_num)],
        device=device,
        dtype=cdtype,
    )
    for i, mq in enumerate(cubic_mag_num):
        if mq == 0:
            matrix[i, lm_mag_num.index(mq)] = 1
        elif mq < 0:
            matrix[i, lm_mag_num.index(mq)] = 1.0j * s2_1
            matrix[i, lm_mag_num.index(-mq)] = 1.0j * s2_1 * (-1) ** (mq + 1)
        else:
            matrix[i, lm_mag_num.index(-mq)] = s2_1
            matrix[i, lm_mag_num.index(mq)] = s2_1 * (-1) ** mq
    return matrix


def create_basis_lm(orbital: str):
    assert orbital in ["s", "p", "d"]
    l_value = ["s", "p", "d"].index(orbital)
    return [[l_value, m, spin] for m in LM_MAG_NUM_DICT[orbital] for spin in [1, -1]]


def _map_lp_sm(lms):
    l_value, m_value, spin = lms
    coef = np.sqrt((l_value - m_value) * (l_value + m_value + 1)) if spin == 1 else 0
    return coef, [l_value, m_value + 1, spin - 2]


def _map_lm_sp(lms):
    l_value, m_value, spin = lms
    coef = np.sqrt((l_value + m_value) * (l_value - m_value + 1)) if spin == -1 else 0
    return coef, [l_value, m_value - 1, spin + 2]


def _map_lz_sz(lms):
    l_value, m_value, spin = lms
    return (m_value if spin == 1 else -m_value), [l_value, m_value, spin]


def get_matrix_lmbasis(basis, device="cpu", dtype=torch.float32):
    ndim = len(basis)
    mat_lp_sm = torch.zeros([ndim, ndim], device=device, dtype=dtype)
    mat_lm_sp = torch.zeros([ndim, ndim], device=device, dtype=dtype)
    mat_lz_sz = torch.zeros([ndim, ndim], device=device, dtype=dtype)
    for i, state in enumerate(basis):
        for matrix, mapper in (
            (mat_lp_sm, _map_lp_sm),
            (mat_lm_sp, _map_lm_sp),
            (mat_lz_sz, _map_lz_sz),
        ):
            coef, mapped = mapper(state)
            if mapped in basis:
                matrix[i, basis.index(mapped)] = coef
    return 0.5 * (mat_lp_sm + mat_lm_sp + mat_lz_sz)


def get_soc_matrix_cubic_basis(
    orbital: str,
    cubic_mag_num=None,
    lm_mag_num=None,
    device="cpu",
    dtype=torch.float32,
):
    assert orbital in ["s", "p", "d"]
    cubic_mag_num = CUBIC_MAG_NUM_DICT[orbital] if cubic_mag_num is None else cubic_mag_num
    lm_mag_num = LM_MAG_NUM_DICT[orbital] if lm_mag_num is None else lm_mag_num
    num_orb = {"s": 1, "p": 3, "d": 5}
    assert len(cubic_mag_num) == num_orb[orbital]
    assert len(lm_mag_num) == num_orb[orbital]

    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        raise TypeError("Only float32 and float64 are supported.")

    lm_basis = create_basis_lm(orbital)
    mtrans = lm2cubic_mat(cubic_mag_num, lm_mag_num, device=device, dtype=dtype)
    msoc_lm = get_matrix_lmbasis(lm_basis, device=device, dtype=dtype)
    msoc_lm_complex = torch.complex(msoc_lm, torch.zeros_like(msoc_lm))

    trans = torch.kron(mtrans, torch.eye(2, device=device, dtype=cdtype)).T
    msoc_cubic = torch.conj(trans.T) @ msoc_lm_complex @ trans

    norbs = len(cubic_mag_num)
    output = torch.zeros([2 * norbs, 2 * norbs], device=device, dtype=cdtype)
    output[0:norbs, 0:norbs] = msoc_cubic[0 : 2 * norbs : 2, 0 : 2 * norbs : 2]
    output[norbs : 2 * norbs, norbs : 2 * norbs] = msoc_cubic[
        1 : 2 * norbs : 2,
        1 : 2 * norbs : 2,
    ]
    output[0:norbs, norbs : 2 * norbs] = msoc_cubic[
        0 : 2 * norbs : 2,
        1 : 2 * norbs : 2,
    ]
    output[norbs : 2 * norbs, 0:norbs] = msoc_cubic[
        1 : 2 * norbs : 2,
        0 : 2 * norbs : 2,
    ]
    return output
