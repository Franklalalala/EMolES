import re
import numpy as np
from argparse import Namespace
from ase import Atoms
import io


BOHR_TO_ANG = 0.529177210903

# 全表（1..118），索引即核电荷数 Z
_ELEM = [
    "", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na","Mg","Al","Si","P", "S", "Cl","Ar","K", "Ca",
    "Sc","Ti","V", "Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y", "Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I", "Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W", "Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U", "Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]


# ========== AO 顺序映射（Gaussian -> PySCF），简洁示例（球谐 5D/7F） ==========

CONV_KEY = 'back2pyscf'
convention_dict = {
    CONV_KEY: Namespace(
        # 每个原子的“壳层串”（示例：def2-SVP 典型收缩）
        atom_to_orbitals_map={
            1: 'ssp',      # H：示例
            3: 'ssspp',    # Li：示例
            6: 'sssppd',   # C
            7: 'sssppd',   # N
            8: 'sssppd',   # O
            9: 'sssppd',   # F
        },
        # 每个壳层内部分量的重排（s:1, p:3, d:5, f:7）
        orbital_idx_map={'s':[0], 'p':[0,1,2], 'd':[4,2,0,1,3], 'f':[6,4,2,0,1,3,5]},
        # 若需要分量变号可在此定义（默认全 +1）
        orbital_sign_map={'s':[1], 'p':[1,1,1], 'd':[1,1,1,1,1], 'f':[1,1,1,1,1,1,1]},
        # 每个原子内部的壳层顺序（此处都为恒等次序，可按基组做细化）
        orbital_order_map={
            1:[0,1,2],
            3:[0,1,2,3,4],
            6:[0,1,2,3,4,5],
            7:[0,1,2,3,4,5],
            8:[0,1,2,3,4,5],
            9:[0,1,2,3,4,5],
        },
    )
}

def build_transform_indices(atoms_Z, convention_key=CONV_KEY):
    conv = convention_dict[convention_key]
    orbitals, order = '', []
    for Z in atoms_Z:
        if Z not in conv.atom_to_orbitals_map or Z not in conv.orbital_order_map:
            raise KeyError(f'No AO mapping preset for Z={Z}; extend convention_dict.')
        offset_shells = len(order)
        orbitals += conv.atom_to_orbitals_map[Z]
        order += [i + offset_shells for i in conv.orbital_order_map[Z]]

    idx_blocks, sgn_blocks = [], []
    for orb in orbitals:
        if orb not in conv.orbital_idx_map:
            raise KeyError(f'No orbital_idx_map for orbital {orb}')
        off = sum(len(b) for b in idx_blocks)
        idx_blocks.append(np.array(conv.orbital_idx_map[orb]) + off)
        sgn_blocks.append(np.array(conv.orbital_sign_map[orb]))
    idx = np.concatenate([idx_blocks[i] for i in order]).astype(np.int32)
    sgn = np.concatenate([sgn_blocks[i] for i in order]).astype(np.int8)
    return idx, sgn

def matrix_transform(M, atoms_Z, convention=CONV_KEY):
    """对称矩阵或批量对称矩阵的 AO 重排 + 变号"""
    M = np.asarray(M)
    idx, sgn = build_transform_indices(atoms_Z, convention_key=convention)
    M2 = M[..., idx, :][..., :, idx]
    M2 = M2 * sgn[:, None]
    M2 = M2 * sgn[None, :]
    return M2

def coeff_transform(C, atoms_Z, convention=CONV_KEY):
    """系数矩阵 C 的 AO 行重排 + 变号（C -> P C）"""
    C = np.asarray(C)
    idx, sgn = build_transform_indices(atoms_Z, convention_key=convention)
    return C[idx, :] * sgn[:, None]

# ========== 数学构造 ==========
def make_density_from_mo(mo_a, nalpha, mo_b=None, nbeta=None):
    """闭壳层返回总密度；开壳层返回 (Da, Db)。"""
    if nbeta is None or nalpha == nbeta:
        Co = mo_a[:, :int(nalpha)]
        return 2.0 * (Co @ Co.T)
    assert mo_b is not None and nbeta is not None
    Coa = mo_a[:, :int(nalpha)]
    Cob = mo_b[:, :int(nbeta)]
    return Coa @ Coa.T, Cob @ Cob.T

def fock_from_mo(S, C, eig):
    """F = S C diag(e) C^T S；要求 C^T S C = I 且 nmo == nbf（完整空间）"""
    return S @ (C @ (np.diag(eig) @ (C.T @ S)))


def overlap_from_orthonormal(X, rcond=1e-10, return_pinv=False):
    """
    稳健从 Orthonormal 基 X（AO->OAO；可为 nbf x nif，nbf>=nif）重构 AO 重叠 S。
    使用 SVD 伪逆，S = X^{+T} X^+，秩 = rank(X)。
    返回:
      - 仅 S（默认）
      - 若 return_pinv=True，同时返回 X^+（OAO->AO 的最小范数映射）
    """
    X = np.asarray(X)
    n, m = X.shape
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0:
        raise ValueError("orthonormal_basis is empty.")
    cutoff = rcond * s.max()
    keep = s > cutoff
    if not np.any(keep):
        raise ValueError("orthonormal_basis is rank-deficient under rcond; no singular values kept.")
    U1 = U[:, keep]             # n x r
    s1 = s[keep]                # r
    V1 = Vt.T[:, keep]          # m x r

    inv_s_sq = (1.0 / s1) ** 2
    S = (U1 * inv_s_sq) @ U1.T
    S = 0.5 * (S + S.T)

    if not return_pinv:
        return S
    # X^+ = V diag(1/s) U^T（截断后）
    X_plus = V1 @ (U1.T / s1[:, None])  # (m x r) @ (r x n) = m x n
    return S, X_plus


def nuc2elem(z):
    return _ELEM[z] if 0 <= z < len(_ELEM) else "Xx"

def _tail_int(line):
    # A49,2X,I10 风格的安全解析：优先取行后半，兜底取最后一个整数
    tail = line[49:] if len(line) > 49 else line
    m = re.findall(r'[-+]?\d+', tail)
    if m:
        return int(m[-1])
    m = re.findall(r'[-+]?\d+', line)
    return int(m[-1]) if m else None

def _parse_count_from_header_line(line):
    m = re.search(r'N\s*=\s*(\d+)', line)
    if m:
        return int(m.group(1))
    return _tail_int(line)

def _read_floats_after_current_line(fh, total_needed):
    vals = []
    for line in fh:
        if len(vals) >= total_needed:
            break
        for tok in line.replace('D', 'E').split():
            try:
                vals.append(float(tok))
            except Exception:
                pass
            if len(vals) >= total_needed:
                break
    return vals[:total_needed]

def _read_ints_after_current_line(fh, total_needed):
    vals = []
    for line in fh:
        if len(vals) >= total_needed:
            break
        for tok in line.split():
            if re.fullmatch(r'[-+]?\d+', tok):
                vals.append(int(tok))
                if len(vals) >= total_needed:
                    break
    return vals[:total_needed]


# 1) 轻量 meta 解析：把 nbf、nif 以及常用信息整合在一起（不读大数据块）
def parse_fchk_meta(fchk_path):
    """
    仅扫描表头与标量，返回一个 meta 字典，不读取矩阵/坐标等大块数据。
    返回字段（尽量填充）：
      - natom, nbf
      - charge, multiplicity
      - nalpha, nbeta
      - neig_alpha, neig_beta  （Alpha/Beta Orbital Energies 的个数）
      - ncoeff_mo_alpha, ncoeff_mo_beta （MO 系数总数）
      - nmo_alpha, nmo_beta    （由 ncoeff_mo_* / nbf 推得）
      - nif                    （优先用 nmo_alpha，其次 neig_alpha，再次 nmo_beta/neig_beta）
    """
    meta = {
        'natom': None, 'nbf': None,
        'charge': None, 'multiplicity': None,
        'nalpha': None, 'nbeta': None,
        'neig_alpha': None, 'neig_beta': None,
        'ncoeff_mo_alpha': None, 'ncoeff_mo_beta': None,
        'nmo_alpha': None, 'nmo_beta': None,
        'nif': None
    }

    # 单次流式扫描
    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            s = line.strip()

            if 'Number of atoms' in line and meta['natom'] is None:
                meta['natom'] = _tail_int(line)

            if 'Number of basis functions' in line and meta['nbf'] is None:
                meta['nbf'] = _tail_int(line)

            if line.startswith('Charge') and meta['charge'] is None:
                meta['charge'] = _tail_int(line)

            if line.startswith('Multiplicity') and meta['multiplicity'] is None:
                meta['multiplicity'] = _tail_int(line)

            if 'Alpha electrons' in line and meta['nalpha'] is None:
                meta['nalpha'] = _tail_int(line)

            if 'Beta electrons' in line and meta['nbeta'] is None:
                meta['nbeta'] = _tail_int(line)

            if line.startswith('Alpha Or') and meta['neig_alpha'] is None:
                meta['neig_alpha = _'] = line.strip()  # 可选调试
                meta['neig_alpha'] = _parse_count_from_header_line(line)

            if line.startswith('Beta Orb') and meta['neig_beta'] is None:
                meta['neig_beta'] = _parse_count_from_header_line(line)

            if line.startswith('Alpha MO') and meta['ncoeff_mo_alpha'] is None:
                meta['ncoeff_mo_alpha'] = _parse_count_from_header_line(line)

            if line.startswith('Beta MO') and meta['ncoeff_mo_beta'] is None:
                meta['ncoeff_mo_beta'] = _parse_count_from_header_line(line)

    # 推导 nmo_alpha/nmo_beta
    nbf = meta['nbf']
    if nbf:
        if meta['ncoeff_mo_alpha'] and meta['ncoeff_mo_alpha'] % nbf == 0:
            meta['nmo_alpha'] = meta['ncoeff_mo_alpha'] // nbf
        if meta['ncoeff_mo_beta'] and meta['ncoeff_mo_beta'] % nbf == 0:
            meta['nmo_beta'] = meta['ncoeff_mo_beta'] // nbf

    # 给出一个“统一 nif”（优先级：Alpha MO -> Alpha eig -> Beta MO -> Beta eig）
    for key in ('nmo_alpha', 'neig_alpha', 'nmo_beta', 'neig_beta'):
        if meta.get(key):
            meta['nif'] = meta[key]
            break

    return meta


# 2) 元素符号 + 核电荷 + 坐标（Bohr->Å），以及 charge、multiplicity
def read_elem_and_coor_from_fchk(fchk_path, natom=None, return_unit='angstrom'):
    """
    返回 (elem, nuc, coor, charge, multiplicity)
      - elem: list[str], 长度 natom
      - nuc:  np.ndarray (natom,), 核电荷 Z
      - coor: np.ndarray (natom, 3), 单位默认为 Å（可设 return_unit='bohr'）
      - charge, multiplicity: int
    仅做流式读取，遇到目标表头就读够并结束。
    """
    elem = None
    nuc = None
    coor = None
    charge = None
    mult = None

    # 如果没传 natom，尽量先轻量解析一下
    if natom is None:
        with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if 'Number of atoms' in line:
                    natom = _tail_int(line)
                    break

    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        while True:
            line = fh.readline()
            if not line:
                break

            if line.startswith('Charge') and charge is None:
                charge = _tail_int(line)
                # 多数 fchk 下一行就是 Multiplicity，但也兼容散落位置
                # 尝试先看下一行，不行就继续主循环里找
                pos = fh.tell()
                next_line = fh.readline()
                if next_line and next_line.startswith('Multiplicity'):
                    mult = _tail_int(next_line)
                else:
                    fh.seek(pos)

            elif line.startswith('Multiplicity') and mult is None:
                mult = _tail_int(line)

            elif line.startswith('Atomic numbers') and nuc is None:
                # 解析 natom（若未确定）
                n_needed = _parse_count_from_header_line(line)
                if n_needed is None:
                    if natom is None:
                        raise ValueError("无法确定 natom。")
                    n_needed = natom
                else:
                    if natom is None:
                        natom = n_needed
                    elif n_needed != natom:
                        raise ValueError(f"natom 不一致: {n_needed} vs {natom}")

                ints = _read_ints_after_current_line(fh, n_needed)
                if len(ints) != n_needed:
                    raise ValueError("读取 Atomic numbers 失败。")
                nuc = np.array(ints, dtype=int)

            elif line.startswith('Current cart') and coor is None:
                n_needed = _parse_count_from_header_line(line)
                if n_needed is None:
                    if natom is None:
                        raise ValueError("无法确定坐标长度（未知 natom）。")
                    n_needed = 3 * natom
                else:
                    # 若借此首次知晓 natom，也自动设置
                    if natom is None and n_needed % 3 == 0:
                        natom = n_needed // 3

                floats = _read_floats_after_current_line(fh, n_needed)
                if len(floats) != n_needed:
                    raise ValueError("读取坐标失败。")
                coor = np.asarray(floats, dtype=float).reshape(-1, 3, order='C')
                if return_unit.lower().startswith('ang'):
                    coor = coor * BOHR_TO_ANG  # Bohr -> Å

            # 读齐就可以结束
            if nuc is not None and coor is not None and charge is not None and mult is not None:
                break

    if nuc is None or coor is None:
        raise ValueError("未成功读取核电荷或坐标。")

    # 生成元素符号
    elem = [nuc2elem(int(z)) for z in nuc]
    return elem, nuc, coor, int(charge if charge is not None else 0), int(mult if mult is not None else 1)


def read_eigenvalues_from_fchk(fchk_path, nif, ab='A'):
    """
    读取 Alpha/Beta Orbital Energies（或 NOON）。
    仅在读数据前扫到关键行即 break；不再在函数内解析 nif。
    返回 (eigs, success)
    """
    prefix = 'Alpha Or' if str(ab).lower().startswith('a') else 'Beta Orb'
    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        # 扫描到表头即停止
        for line in fh:
            if line.startswith(prefix):
                # 表头数检查（若存在）
                cnt = _parse_count_from_header_line(line)
                if cnt is not None and cnt != int(nif):
                    return None, False
                # 紧接着读 nif 个数值
                vals = _read_floats_after_current_line(fh, nif)
                if len(vals) != nif:
                    return None, False
                return np.asarray(vals, dtype=np.float64), True
    return None, False

def read_dm_from_fchk(fchk_path, itype, nbf):
    """
    读取各种 AO 密度矩阵（上三角压缩），依赖外部给定 nbf。
    itype: 1~12（与 Fortran 一致）
    返回 (dm, success)
    """
    dm_keys = [
        'Total SCF D', 'Spin SCF De',
        'Total CI De', 'Spin CI Den',
        'Total MP2 D', 'Spin MP2 De',
        'Total CC De', 'Spin CC Den',
        'Total CI Rh', 'Spin CI Rho',
        'Total 2nd O', 'Spin 2nd Or',
    ]
    if not (1 <= itype <= 12):
        return None, False
    need = nbf * (nbf + 1) // 2
    prefix = dm_keys[itype - 1]

    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if line.startswith(prefix):
                cnt = _parse_count_from_header_line(line)
                if cnt is None or cnt != need:
                    return None, False
                vals = _read_floats_after_current_line(fh, need)
                if len(vals) != need:
                    return None, False
                # 还原上三角为全矩阵，并对称化
                dm = np.zeros((nbf, nbf), dtype=np.float64)
                k = 0
                for i in range(nbf):
                    for j in range(i + 1):
                        dm[j, i] = vals[k]
                        k += 1
                dm = dm + dm.T - np.diag(np.diag(dm))
                return dm, True
    return None, False

def read_mo_from_fchk(fchk_path, nbf, nif, ab='A'):
    """
    读取 Alpha/Beta MO 系数矩阵，参数 nbf、nif 必须外部提供。
    返回 (mo, success)，mo 形状 (nbf, nif)，Fortran 列主序。
    """
    prefix = 'Alpha MO' if str(ab).lower().startswith('a') else 'Beta MO'
    need = nbf * nif

    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if line.startswith(prefix):
                cnt = _parse_count_from_header_line(line)
                if cnt is None or cnt != need:
                    return None, False
                vals = _read_floats_after_current_line(fh, need)
                if len(vals) != need:
                    return None, False
                mo = np.asarray(vals, dtype=np.float64).reshape((nbf, nif), order='F')
                return mo, True
    return None, False


def read_orthonormal_basis_from_fchk(fchk_path, nbf=None, label_prefix='Orthonormal'):
    """
    读取 Gaussian .fchk 文件中的 Orthonormal 基（OAO）矩阵。
    - 自动从文件头解析 Number of basis functions；也可手动传 nbf。
    - 默认匹配以 'Orthonormal' 开头的标签行（与 Fortran 示例一致）。
    - 返回 (mo, success)：mo 为 (nbf, nbf) 的 float64 矩阵；若失败则 (None, False)。
    """
    with open(fchk_path, 'r') as fh:
        lines = fh.readlines()

    # 若未提供 nbf，尝试从文件头解析
    if nbf is None:
        for ln in lines[:300]:  # 头部通常就有
            if 'Number of basis functions' in ln:
                m = re.search(r'Number of basis functions\s+[A-Z]?\s+(\d+)', ln)
                if m:
                    nbf = int(m.group(1))
                else:
                    ints = re.findall(r'\d+', ln)
                    if ints:
                        nbf = int(ints[-1])
                break

    # 找到 Orthonormal 开头的标签行
    label_idx = None
    total_needed = None
    for i, ln in enumerate(lines):
        if ln.startswith(label_prefix):
            label_idx = i
            # 若标签行上有 N=，也读出来
            m = re.search(r'N\s*=\s*(\d+)', ln)
            if m:
                total_needed = int(m.group(1))
            break

    if label_idx is None:
        return None, False

    # 推断 nbf 或 total_needed
    if nbf is None and total_needed is not None:
        rt = int(round(total_needed ** 0.5))
        if rt * rt == total_needed:
            nbf = rt
    if nbf is None:
        return None, False
    if total_needed is None:
        total_needed = nbf * nbf

    # 读取数据（处理 D 指数）
    vals = []
    for ln in lines[label_idx + 1:]:
        if len(vals) >= total_needed:
            break
        for tok in ln.replace('D', 'E').split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if len(vals) >= total_needed:
            break

    if len(vals) < total_needed:
        return None, False

    # 按 Fortran 顺序重塑，保持与 Fortran 读入一致的列主序
    mo = np.array(vals[:total_needed], dtype=np.float64).reshape((nbf, nbf), order='F')
    return mo, True


def read_fchk(fchk_path,
              read_matrices=['density', 'orthonormal_basis', 'mo', 'eigenvalues'],
              return_unit='angstrom'):
    """
    一次性读取 fchk 文件，返回 ASE Atoms 对象，包含所有 meta 和矩阵信息。

    参数:
        fchk_path: str, fchk 文件路径
        read_matrices: list, 要读取的矩阵，可选项:
            - 'density': 读取 Total SCF Density (和 Spin SCF Density if UKS)
            - 'orthonormal_basis': 读取 Orthonormal basis
            - 'mo': 读取 MO coefficients (Alpha and Beta if UKS)
            - 'eigenvalues': 读取 orbital energies
        return_unit: str, 坐标单位 ('angstrom' 或 'bohr')

    返回:
        ase.Atoms 对象，其中 info 字典包含:
            - meta: 所有元信息（包括泛函、基组等）
            - matrices: 字典，包含请求的矩阵
            - convention: 轨道约定信息
    """

    # 用于存储所有数据
    data = {
        'meta': {},
        'matrices': {},
        'atoms_data': {},
        'functional_basis': {}
    }

    # 一次读取整个文件，使用流式处理
    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()

    # 解析第二行的泛函和基组信息
    if len(lines) > 1:
        second_line = lines[1]
        # 格式通常是: "计算类型      泛函                                                      基组"
        parts = second_line.split()
        if len(parts) >= 2:
            # 提取计算类型（如 FOpt, SP等）
            calc_type = parts[0] if parts else ''
            data['functional_basis']['calculation_type'] = calc_type

            # 提取泛函（如 RB3LYP, UB3LYP等）
            functional = parts[1] if len(parts) > 1 else ''
            # 去掉前缀 R/U/RO
            if functional.startswith(('R', 'U', 'RO')):
                if functional.startswith('RO'):
                    data['functional_basis']['reference'] = 'RO'
                    functional = functional[2:]
                else:
                    data['functional_basis']['reference'] = functional[0]
                    functional = functional[1:]
            data['functional_basis']['functional'] = functional

            # 提取基组（通常在行末）
            # 基组通常在第二行的最后，可能有空格分隔
            basis_part = second_line[60:].strip() if len(second_line) > 60 else ''
            if not basis_part and len(parts) > 2:
                basis_part = parts[-1]
            data['functional_basis']['basis_set'] = basis_part

    current_section = None
    current_count = 0
    current_data = []

    for i, line in enumerate(lines):
        # === Meta 信息解析 ===
        if 'Number of atoms' in line and 'natom' not in data['meta']:
            data['meta']['natom'] = _tail_int(line)

        elif 'Number of basis functions' in line and 'nbf' not in data['meta']:
            data['meta']['nbf'] = _tail_int(line)

        elif line.startswith('Charge') and 'charge' not in data['meta']:
            data['meta']['charge'] = _tail_int(line)

        elif line.startswith('Multiplicity') and 'multiplicity' not in data['meta']:
            data['meta']['multiplicity'] = _tail_int(line)

        elif 'Number of alpha electrons' in line or 'Alpha electrons' in line:
            if 'nalpha' not in data['meta']:
                data['meta']['nalpha'] = _tail_int(line)

        elif 'Number of beta electrons' in line or 'Beta electrons' in line:
            if 'nbeta' not in data['meta']:
                data['meta']['nbeta'] = _tail_int(line)

        elif 'Number of contracted shells' in line:
            data['meta']['nshells'] = _tail_int(line)

        elif 'Number of primitive shells' in line:
            data['meta']['nprimitive'] = _tail_int(line)

        elif 'Highest angular momentum' in line:
            data['meta']['max_angular_momentum'] = _tail_int(line)

        elif 'Largest degree of contraction' in line:
            data['meta']['max_contraction'] = _tail_int(line)

        elif 'Pure/Cartesian d shells' in line:
            val = _tail_int(line)
            data['meta']['cartesian_d'] = (val == 1)

        elif 'Pure/Cartesian f shells' in line:
            val = _tail_int(line)
            data['meta']['cartesian_f'] = (val == 1)

        # === 原子数据 ===
        elif line.startswith('Atomic numbers'):
            current_section = 'atomic_numbers'
            current_count = _parse_count_from_header_line(line) or data['meta'].get('natom', 0)
            current_data = []

        elif line.startswith('Current cartesian coordinates'):
            current_section = 'coordinates'
            current_count = _parse_count_from_header_line(line) or (data['meta'].get('natom', 0) * 3)
            current_data = []

        # === Shell 信息（用于 convention） ===
        elif line.startswith('Shell types'):
            current_section = 'shell_types'
            current_count = _parse_count_from_header_line(line) or data['meta'].get('nshells', 0)
            current_data = []

        elif line.startswith('Shell to atom map'):
            current_section = 'shell_to_atom'
            current_count = _parse_count_from_header_line(line) or data['meta'].get('nshells', 0)
            current_data = []

        # === 矩阵数据（根据 read_matrices 参数） ===
        elif 'density' in read_matrices and line.startswith('Total SCF Density'):
            current_section = 'total_scf_density'
            nbf = data['meta'].get('nbf', 0)
            current_count = nbf * (nbf + 1) // 2
            current_data = []

        elif 'density' in read_matrices and line.startswith('Spin SCF Density'):
            current_section = 'spin_scf_density'
            nbf = data['meta'].get('nbf', 0)
            current_count = nbf * (nbf + 1) // 2
            current_data = []

        elif 'orthonormal_basis' in read_matrices and line.startswith('Orthonormal'):
            current_section = 'orthonormal_basis'
            current_count = _parse_count_from_header_line(line)
            current_data = []

        elif 'mo' in read_matrices and line.startswith('Alpha MO coefficients'):
            current_section = 'alpha_mo'
            current_count = _parse_count_from_header_line(line)
            data['meta']['ncoeff_mo_alpha'] = current_count
            current_data = []

        elif 'mo' in read_matrices and line.startswith('Beta MO coefficients'):
            current_section = 'beta_mo'
            current_count = _parse_count_from_header_line(line)
            data['meta']['ncoeff_mo_beta'] = current_count
            current_data = []

        elif 'eigenvalues' in read_matrices and line.startswith('Alpha Orbital Energies'):
            current_section = 'alpha_eigenvalues'
            current_count = _parse_count_from_header_line(line)
            data['meta']['neig_alpha'] = current_count
            current_data = []

        elif 'eigenvalues' in read_matrices and line.startswith('Beta Orbital Energies'):
            current_section = 'beta_eigenvalues'
            current_count = _parse_count_from_header_line(line)
            data['meta']['neig_beta'] = current_count
            current_data = []

        # === 数据读取 ===
        elif current_section is not None and current_count > 0:
            # 根据section类型决定如何解析
            if current_section in ['atomic_numbers', 'shell_types', 'shell_to_atom']:
                # 整数解析
                for tok in line.split():
                    if re.fullmatch(r'[-+]?\d+', tok):
                        current_data.append(int(tok))
                        if len(current_data) >= current_count:
                            # 存储完整数据
                            data['atoms_data'][current_section] = np.array(current_data[:current_count])
                            current_section = None
                            current_count = 0
                            current_data = []
                            break
            else:
                # 浮点数解析
                for tok in line.replace('D', 'E').split():
                    try:
                        current_data.append(float(tok))
                        if len(current_data) >= current_count:
                            # 存储完整数据
                            if current_section == 'coordinates':
                                coords = np.array(current_data[:current_count]).reshape(-1, 3)
                                if return_unit.lower().startswith('ang'):
                                    coords *= BOHR_TO_ANG
                                data['atoms_data'][current_section] = coords
                            elif current_section in ['total_scf_density', 'spin_scf_density']:
                                # 将上三角展开为完整矩阵
                                nbf = data['meta'].get('nbf', 0)
                                dm = np.zeros((nbf, nbf))
                                k = 0
                                for ii in range(nbf):
                                    for jj in range(ii + 1):
                                        dm[jj, ii] = current_data[k]
                                        k += 1
                                dm = dm + dm.T - np.diag(np.diag(dm))
                                data['matrices'][current_section] = dm
                            elif current_section == 'orthonormal_basis':
                                nbf = data['meta'].get('nbf', 0)
                                nmo = current_count // nbf
                                data['matrices'][current_section] = np.array(
                                    current_data[:current_count]).reshape((nbf, nmo), order='F')
                            elif current_section in ['alpha_mo', 'beta_mo']:
                                nbf = data['meta'].get('nbf', 0)
                                nmo = current_count // nbf if nbf > 0 else 0
                                if nmo > 0:
                                    key = 'nmo_alpha' if 'alpha' in current_section else 'nmo_beta'
                                    data['meta'][key] = nmo
                                    data['matrices'][current_section] = np.array(
                                        current_data[:current_count]).reshape((nbf, nmo), order='F')


                            elif current_section in ['alpha_eigenvalues', 'beta_eigenvalues']:
                                data['matrices'][current_section] = np.array(current_data[:current_count])
                            else:
                                data['matrices'][current_section] = np.array(current_data[:current_count])

                            current_section = None
                            current_count = 0
                            current_data = []
                            break
                    except ValueError:
                        pass

    # 后处理 meta 信息
    nbf = data['meta'].get('nbf')
    if nbf:
        for key in ['nmo_alpha', 'nmo_beta']:
            coeff_key = f"ncoeff_mo_{key.split('_')[1]}"
            if coeff_key in data['meta'] and data['meta'][coeff_key] % nbf == 0:
                data['meta'][key] = data['meta'][coeff_key] // nbf

    # 确定 nif
    for key in ['nmo_alpha', 'neig_alpha', 'nmo_beta', 'neig_beta']:
        if data['meta'].get(key):
            data['meta']['nif'] = data['meta'][key]
            break

    # 获取轨道约定信息（如果有 shell 信息的话）
    convention = None
    try:
        convention = get_convention(fchk_path)
    except Exception as e:
        print(f"Warning: Could not get convention info: {e}")

    # 创建 ASE Atoms 对象
    if 'atomic_numbers' in data['atoms_data'] and 'coordinates' in data['atoms_data']:
        symbols = [nuc2elem(int(z)) for z in data['atoms_data']['atomic_numbers']]
        positions = data['atoms_data']['coordinates']

        atoms = Atoms(
            symbols=symbols,
            positions=positions
        )

        # 将所有信息存储在 info 字典中
        atoms.info['fchk_meta'] = data['meta']
        atoms.info['fchk_matrices'] = data['matrices']
        atoms.info['charge'] = data['meta'].get('charge', 0)
        atoms.info['multiplicity'] = data['meta'].get('multiplicity', 1)
        atoms.info['nuclear_charges'] = data['atoms_data']['atomic_numbers']
        atoms.info['functional_basis'] = data['functional_basis']

        # 添加 convention 信息
        if convention:
            atoms.info['convention'] = convention

        # 添加其他有用的 shell 信息
        if 'shell_types' in data['atoms_data']:
            atoms.info['shell_types'] = data['atoms_data']['shell_types']
        if 'shell_to_atom' in data['atoms_data']:
            atoms.info['shell_to_atom'] = data['atoms_data']['shell_to_atom']

        return atoms
    else:
        raise ValueError("无法从 fchk 文件中读取原子信息")


def get_convention(fchk_path):
    """
    解析 fchk 文件，获取基组和轨道排列信息

    返回:
        dict: 包含以下键值
            - atom_to_simplified_orbitals: {元素符号: 简化轨道串如"sssppd"}
            - atom_to_dftio_orbitals: {元素符号: DFTIO格式如"3s2p1d"}
            - atom_to_transform_indices: {元素符号: Gaussian->PySCF的AO索引映射列表}
    """

    # Shell type 到轨道类型的映射
    shell_type_to_orbital = {
        0: 's',  # s orbital
        1: 'p',  # p orbital (3 components)
        -1: 'p',  # p orbital (spherical, 3 components)
        2: 'd',  # d orbital (Cartesian, 6 components)
        -2: 'd',  # d orbital (spherical, 5 components)
        3: 'f',  # f orbital (Cartesian, 10 components)
        -3: 'f',  # f orbital (spherical, 7 components)
    }

    # 读取必要信息
    natom = None
    atomic_numbers = []
    nshells = None
    shell_types = []
    shell_to_atom = []
    cart_d = False  # 是否使用笛卡尔d轨道
    cart_f = False  # 是否使用笛卡尔f轨道

    with open(fchk_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # 读取原子数
        if 'Number of atoms' in line and natom is None:
            natom = _tail_int(line)

        # 读取原子序数
        elif line.startswith('Atomic numbers') and not atomic_numbers:
            n = _parse_count_from_header_line(line) or natom
            # 读取后续行的整数
            j = i + 1
            while len(atomic_numbers) < n and j < len(lines):
                for tok in lines[j].split():
                    if tok.isdigit() or (tok.startswith('-') and tok[1:].isdigit()):
                        atomic_numbers.append(int(tok))
                        if len(atomic_numbers) >= n:
                            break
                j += 1

        # 读取收缩壳层数
        elif 'Number of contracted shells' in line and nshells is None:
            nshells = _tail_int(line)

        # 读取壳层类型
        elif line.startswith('Shell types') and not shell_types:
            n = _parse_count_from_header_line(line) or nshells
            j = i + 1
            while len(shell_types) < n and j < len(lines):
                for tok in lines[j].split():
                    if tok.lstrip('-').isdigit():
                        val = int(tok)
                        shell_types.append(val)
                        # 检查是否使用笛卡尔基函数
                        if val == 2:
                            cart_d = True
                        elif val == 3:
                            cart_f = True
                        if len(shell_types) >= n:
                            break
                j += 1

        # 读取壳层到原子的映射
        elif 'Shell to atom map' in line and not shell_to_atom:
            n = _parse_count_from_header_line(line) or nshells
            j = i + 1
            while len(shell_to_atom) < n and j < len(lines):
                for tok in lines[j].split():
                    if tok.isdigit():
                        shell_to_atom.append(int(tok))
                        if len(shell_to_atom) >= n:
                            break
                j += 1

        # 读取 Pure/Cartesian d/f shells 信息
        elif 'Pure/Cartesian d shells' in line:
            val = _tail_int(line)
            if val == 1:
                cart_d = True
        elif 'Pure/Cartesian f shells' in line:
            val = _tail_int(line)
            if val == 1:
                cart_f = True

    if not atomic_numbers or not shell_types or not shell_to_atom:
        raise ValueError("Failed to parse shell information from fchk file")

    # 将原子序数转换为元素符号
    elements = [nuc2elem(z) for z in atomic_numbers]

    # 为每个原子收集其壳层信息
    atom_shells = {i + 1: [] for i in range(natom)}  # shell_to_atom 使用1-based索引
    for shell_idx, atom_idx in enumerate(shell_to_atom):
        shell_type = shell_types[shell_idx]
        orbital_type = shell_type_to_orbital.get(shell_type, '?')
        atom_shells[atom_idx].append(orbital_type)

    # 生成输出
    atom_to_simplified_orbitals = {}
    atom_to_dftio_orbitals = {}
    atom_to_transform_indices = {}

    for atom_idx in range(natom):
        elem = elements[atom_idx]
        shells = atom_shells[atom_idx + 1]  # 1-based

        # 简化轨道串
        simplified = ''.join(shells)
        atom_to_simplified_orbitals[elem] = simplified

        # DFTIO格式（计数每种轨道类型）
        orbital_counts = {}
        for orb in shells:
            orbital_counts[orb] = orbital_counts.get(orb, 0) + 1

        dftio_parts = []
        for orb_type in ['s', 'p', 'd', 'f', 'g']:
            if orb_type in orbital_counts:
                count = orbital_counts[orb_type]
                dftio_parts.append(f"{count}{orb_type}")
        atom_to_dftio_orbitals[elem] = ''.join(dftio_parts)

        # 生成转换索引（Gaussian -> PySCF）
        indices = []
        ao_counter = 0

        for shell in shells:
            if shell == 's':
                # s轨道：1个分量，顺序相同
                indices.append(ao_counter)
                ao_counter += 1

            elif shell == 'p':
                # p轨道：3个分量，顺序相同
                for i in range(3):
                    indices.append(ao_counter + i)
                ao_counter += 3

            elif shell == 'd':
                # d轨道：球谐5个分量
                # Gaussian顺序: d0, d+1, d-1, d+2, d-2 (即 z², xz, yz, x²-y², xy)
                # PySCF顺序: d-2, d-1, d0, d+1, d+2 (即 xy, yz, z², xz, x²-y²)
                # 映射: [2, 3, 4, 1, 0] -> [0, 1, 2, 3, 4]
                gauss_to_pyscf = [4, 2, 0, 1, 3]  # 从Gaussian顺序到PySCF顺序
                for idx in gauss_to_pyscf:
                    indices.append(ao_counter + idx)
                ao_counter += 5

            elif shell == 'f':
                # f轨道：球谐7个分量
                # Gaussian顺序: f0, f+1, f-1, f+2, f-2, f+3, f-3
                # PySCF顺序: f-3, f-2, f-1, f0, f+1, f+2, f+3
                gauss_to_pyscf = [6, 4, 2, 0, 1, 3, 5]
                for idx in gauss_to_pyscf:
                    indices.append(ao_counter + idx)
                ao_counter += 7

        atom_to_transform_indices[elem] = indices

    # 处理重复元素的情况（如两个H原子）
    # 合并相同元素的信息（保持第一个出现的）
    unique_elements = {}
    for elem in elements:
        if elem not in unique_elements:
            unique_elements[elem] = True

    result = {
        "atom_to_simplified_orbitals": {elem: atom_to_simplified_orbitals[elem]
                                        for elem in unique_elements if elem in atom_to_simplified_orbitals},
        "atom_to_dftio_orbitals": {elem: atom_to_dftio_orbitals[elem]
                                   for elem in unique_elements if elem in atom_to_dftio_orbitals},
        "atom_to_transform_indices": {elem: atom_to_transform_indices[elem]
                                      for elem in unique_elements if elem in atom_to_transform_indices}
    }

    return result


# 如果作为主程序运行，执行测试
if __name__ == '__main__':
    fchk = 'gau.fchk'

    # 1) 一次性拿 meta
    meta = parse_fchk_meta(fchk)
    print(meta)
    # 常用字段
    print('nbf =', meta['nbf'], 'nif =', meta['nif'], 'natom =', meta['natom'])
    print('charge =', meta['charge'], 'mult =', meta['multiplicity'])

    # 2) 拿元素、核电荷、坐标（Å）
    elem, nuc, xyz, charge, mult = read_elem_and_coor_from_fchk(fchk)
    print('natom =', len(elem), 'charge =', charge, 'mult =', mult)
    print('first 5 atoms:', list(zip(elem[:5], nuc[:5], xyz[:5])))

    # 如果要 Bohr 坐标
    elem, nuc, xyz_bohr, charge, mult = read_elem_and_coor_from_fchk(fchk, return_unit='bohr')

    # 读 Alpha eigenvalues（或 NOON）
    eigs_a, ok = read_eigenvalues_from_fchk(fchk, nif=meta['nif'], ab='A')
    print('Alpha eigs:', ok, eigs_a.shape if ok else None, eigs_a[:8] if ok else None)

    # 读 Total SCF 密度矩阵（itype=1），需要 nbf
    dm, ok = read_dm_from_fchk(fchk, itype=1, nbf=meta['nbf'])
    print('DM:', ok, dm.shape if ok else None)
    if ok:
        print(dm[:3, :3])

    # 读 Alpha MO（需要 nbf, nif）
    mo_a, ok = read_mo_from_fchk(fchk, nbf=meta['nbf'], nif=meta['nif'], ab='A')
    print('Alpha MO:', ok, mo_a.shape if ok else None)
    if ok:
        print(mo_a[:3, :3])

    print("\n" + "=" * 60)
    print("=" * 60)
    result = get_convention('gau.fchk')
    import json

    print(json.dumps(result, indent=2))