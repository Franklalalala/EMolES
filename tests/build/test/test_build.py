import os
import sys
import numpy as np
from ase import Atoms
from ase.io import write

import emoles.build.patch_picker as pp

# 保存原始函数
_original_get_patch = pp.get_patch_atoms_and_indices


def _patched_get_patch_atoms_and_indices(identifier, *args, **kwargs):
    """
    补丁函数：如果是单个 ASE 原子对象（如 Li+），直接返回，跳过 RDKit 计算。
    """
    # 检查是否为单原子 ASE 对象
    if isinstance(identifier, Atoms) and len(identifier) == 1:
        # 直接返回该原子，Patch索引设为 [0]
        return identifier, [0]

    # 否则调用原始逻辑
    return _original_get_patch(identifier, *args, **kwargs)


# 应用补丁：覆盖库中的函数
pp.get_patch_atoms_and_indices = _patched_get_patch_atoms_and_indices
print(">> [System] Applied Hotfix to patch_picker: Skipping RDKit for single atoms.")
# -----------------------------------------------------------------------------


# 正常导入 build_cluster
from emoles.build.cluster import build_cluster


def run_batch_build():
    print("=========================================================")
    print("   BATCH CLUSTER BUILDING: Li+ with Various Solvents/Salts")
    print("=========================================================")

    # 定义测试案例列表
    # 格式: (Category, Name, SMILES, NumLigands, MaxPatchAtoms)
    test_cases = [
        # --- 1. Novel Ethers (From Paper) ---
        # DPE: Dipropyl ether (Symmetric ether) -> Li bind to O
        ("Novel_Ethers", "Li_4DPE", "CCCOCCC", 4, 1),
        # FEME: 2,2-Difluoroethyl methyl ether -> Li bind to O (F should be avoided by tuned weights)
        ("Novel_Ethers", "Li_4FEME", "COCC(F)F", 4, 1),

        # --- 2. Novel Salts (Imidazolide based) ---
        # TDI: 4,5-dicyano-2-(trifluoromethyl)imidazolide
        # Expected binding: N in the ring or CN groups? Usually ring Nitrogens are active.
        # SMILES represents the anion.
        ("Novel_Salts", "Li_TDI", "FC(F)(F)C1=NC(C#N)=C(C#N)[N-]1", 1, 3),

        # --- 3. Carbonates (C=O binding) ---
        ("Carbonates", "Li_1EC", "C1COC(=O)O1", 1, 1),
        ("Carbonates", "Li_4DMC", "COC(=O)OC", 4, 1),
        ("Carbonates", "Li_1FEC", "FC1COC(=O)O1", 1, 1),

        # --- 4. Ethers (C-O-C binding) ---
        ("Ethers", "Li_2DME", "COCCOC", 2, 2),  # Bidentate
        ("Ethers", "Li_4THF", "C1CCOC1", 4, 1),
        ("Ethers", "Li_2DOL", "C1COCO1", 2, 2),
        ("Ethers", "Li_2DX", "C1COCCO1", 2, 2),

        # --- 5. Esters & Lactones ---
        ("Esters", "Li_4EA", "CCOC(C)=O", 4, 1),
        ("Esters", "Li_4gBL", "O=C1CCCO1", 4, 1),

        # --- 6. Sulfur & Nitrogen ---
        ("Others", "Li_4DMSO", "CS(=O)C", 4, 1),
        ("Others", "Li_4AN", "CC#N", 4, 1),
        ("Others", "Li_2TMS", "O=S1(=O)CCCC1", 2, 2),

        # --- 7. Standard Salts / Anions ---
        ("Standard_Salts", "Li_DFOB", "F[B-]1(F)OC(=O)C(=O)O1", 1, 3),
        ("Standard_Salts", "Li_FSI", "O=S(=O)(F)[N-]S(=O)(=O)F", 1, 2),
        ("Standard_Salts", "Li_TFSI", "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", 1, 2),
        ("Standard_Salts", "Li_BF4", "F[B-](F)(F)F", 1, 1),
    ]

    output_root = "Li_Solvation_Test_Results_v2"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 创建 Li 原子对象
    li_atom = Atoms("Li", positions=[[0.0, 0.0, 0.0]])

    for category, name, smiles, num_ligands, max_patch in test_cases:
        print(f"\n>>> Building: {name} ({category})")

        cat_dir = os.path.join(output_root, category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)

        try:
            cluster = build_cluster(
                ion_identifier=li_atom.copy(),
                ligand_molecule_info=[(smiles, num_ligands)],

                # --- 参数设置 ---
                relative_score_threshold=0.85,  # 严格的筛选门槛
                max_patch_atoms=max_patch,  # 动态最大配位点数
                initial_sphere_skin_factor=0.75,  # 紧凑初始堆积
                sphere_skin_increment_factor=0.02,  # 缓慢膨胀
                target_no_clashes=True,
                rotation_opt_iterations=50,
                verbose=False,
                initial_ligand_orientation="aligned_to_ion"
            )

            filename = os.path.join(cat_dir, f"{name}.xyz")
            write(filename, cluster)
            print(f"    [SUCCESS] Saved to {filename}")

        except Exception as e:
            print(f"    [FAILED] {e}")
            import traceback
            traceback.print_exc()

    print("\nBatch processing complete. Check folder:", output_root)


if __name__ == "__main__":
    run_batch_build()