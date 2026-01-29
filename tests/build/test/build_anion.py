#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Li+ 阴离子配位结构构建
- BF4/PF6: SMILES（底层兜底）
- 其他阴离子: DB 3D 坐标 + 正确电荷
- ClO4/NO3: 额外补充
"""

import os
import numpy as np
import ase.db
from ase import Atoms
from ase.io import write
from rdkit import RDLogger

# 禁用 RDKit 的冗余日志
RDLogger.DisableLog('rdApp.*')

from emoles.build.cluster import build_cluster

# =============================================================================
# 配置
# =============================================================================

# BF4/PF6: 用 SMILES（拓扑问题，底层已兜底，会自动识别为 Complex Anion）
SMILES_FALLBACK = {
}

# 其他阴离子: 用 DB 3D 坐标，但需指定电荷
# 底层 _parse_input 会读取这些电荷并传递给 DetermineBonds
ANION_CHARGES = {
    "FSI": -1,
    "TFSI": -1,
    "TDI": -1,
    "DFOB": -1,
    "PO2F2": -1,
    "FTFSI": -1,
    "BOB": -1,
    "Tf": -1,
}

# 不在数据库中
EXTRA_ANIONS = {
    "ClO4": "[O-][Cl](=O)(=O)=O",
    "NO3": "[O-][N+](=O)[O-]",
}


# =============================================================================
# 核心函数
# =============================================================================

def process_database(db_path, output_folder, ligand_count=1, max_patch_atoms=3):
    if not os.path.exists(db_path):
        print(f"!! DB not found: {db_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    db = ase.db.connect(db_path)
    li = Atoms("Li", positions=[[0.0, 0.0, 0.0]])

    print(f"\n>>> Processing {db_path} ({db.count()} entries)")

    ok, total = 0, 0
    for row in db.select():
        total += 1
        name = getattr(row, 'name', f"mol_{row.id}")
        base = name.split('_')[0]

        # 决定输入方式
        if base in SMILES_FALLBACK:
            mol_in = SMILES_FALLBACK[base]
            src = "SMILES"
        else:
            mol_in = row.toatoms()

            # 关键：设置电荷让底层能正确推断化学键
            # 新版 patch_picker._parse_input 会自动读取这里的 initial_charges
            if base in ANION_CHARGES:
                charge = -1
                mol_in.set_initial_charges(np.full(len(mol_in), charge / len(mol_in)))
                src = f"DB_3D (q={charge})"
            else:
                src = "DB_3D"

        try:
            cluster = build_cluster(
                ion_identifier=li.copy(),
                ligand_molecule_info=[(mol_in, ligand_count)],
                relative_score_threshold=0.80,
                max_patch_atoms=max_patch_atoms,
                initial_sphere_skin_factor=0.8,
                sphere_skin_increment_factor=0.05,
                target_no_clashes=True,
                rotation_opt_iterations=40,
                verbose=False,
                initial_ligand_orientation="aligned_to_ion"
            )
            write(os.path.join(output_folder, f"Li_{ligand_count}_{name}.xyz"), cluster)
            ok += 1
            print(f"  [OK] {name} ({src})")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"<< Done: {ok}/{total}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Li+ Anion Cluster Builder")
    print("=" * 50)

    process_database("anion.db", "anion", ligand_count=1, max_patch_atoms=3)