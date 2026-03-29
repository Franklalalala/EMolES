import os
import random
import json
import numpy as np

from ase import Atoms
from ase.db import connect
from ase.io import write

# 直接从更新后的底层库导入我们需要的函数和配置
from emoles.build.func_sub import (
    make_frag_library_default,
    FilterConfig,
    SubstituteConfig,
    clean_filter_atoms,
    substitute_once_rigid,
)


# =====================================================================
# 遍历测试执行代码 (匹配全新的 Rigid Body 引擎)
# =====================================================================
def test_exhaustive_with_rigid_body(
        ea_db_path: str = "ea.db",
        out_dir: str = "rigid_sub_test_xyz",
        base_id: int | None = 1,
        target_atom_index: int = None
):
    os.makedirs(out_dir, exist_ok=True)

    db = connect(ea_db_path)
    if base_id is None:
        all_ids = [row.id for row in db.select()]
        base_id = random.Random(0).choice(all_ids)

    base_atoms = db.get(id=base_id).toatoms()

    if target_atom_index is None:
        h_indices = [atom.index for atom in base_atoms if atom.symbol == 'H']
        if not h_indices:
            raise ValueError("Base 分子中没有找到 H 原子！")
        target_atom_index = h_indices[-1]

    print(
        f"\n[INFO] 使用完美刚体对齐引擎 | Base ID: {base_id} | 锁定替换位点: Index {target_atom_index} ({base_atoms[target_atom_index].symbol})")

    full_frag_lib = make_frag_library_default(seed=42)
    print(f"[INFO] 成功加载 {len(full_frag_lib)} 种官能团 (包含纯砜类对照组 SO2CH3)。")

    # 配置规则：严格剔除小环与错误化学键
    cfg = FilterConfig(
        forbid_oo_bond=True,
        forbid_oh_bond=True,
        forbid_cc_triple=True,
        forbid_3_4_member_rings=True,
        reject_rdkit_warnings=True,
    )

    # 配置拼接器：步长 10度，开启多态随机性
    sub_cfg = SubstituteConfig(
        dihedral_step=10,
        stochastic_dihedral=True
    )

    rng = random.Random(42)

    summary = []

    for frag_name in full_frag_lib.keys():
        print(f"\n-> 正在测试装载: {frag_name}")

        try:
            # 传入了最新的 sub_cfg 和 rng
            new_atoms, meta = substitute_once_rigid(
                mol=base_atoms,
                h_idx=target_atom_index,
                group_name=frag_name,
                frag_lib=full_frag_lib,
                filter_cfg=cfg,
                sub_cfg=sub_cfg,
                rng=rng
            )

            if new_atoms is not None and meta.get("ok"):
                passed, smiles, inchi, reason = clean_filter_atoms(new_atoms, cfg)

                if passed:
                    fn = os.path.join(out_dir, f"rigid_site{target_atom_index}_{frag_name}.xyz")
                    write(fn, new_atoms)

                    final_data = meta.get("final", {})
                    angle = final_data.get("dihedral_angle", -1)
                    min_dist = final_data.get("min_nonbonded_dist", -1)
                    ideal_bond = final_data.get("bond_length_ideal", -1)
                    actual_bond = final_data.get("bond_length_actual", -1)

                    print(f"   [OK] 刚体拼接成功！")
                    print(f"        - 最佳二面角: {angle}°")
                    print(f"        - 最小非键间距: {min_dist:.3f} Å")

                    if abs(ideal_bond - actual_bond) > 1e-4:
                        print(
                            f"        - ⚠ 发生位阻外推拉伸: 理想键长 {ideal_bond:.3f} Å -> 实际键长 {actual_bond:.3f} Å")
                    else:
                        print(f"        - 成键状态完美: 键长 {actual_bond:.3f} Å")

                    print(f"        - 已写入: {os.path.basename(fn)}")

                    summary.append({
                        "frag": frag_name, "smiles": smiles,
                        "min_nonbonded_dist": min_dist, "actual_bond": actual_bond
                    })
                else:
                    print(f"   [Fail] 几何无严重碰撞，但在 RDKit 化学拓扑验证中被拦截: {reason}")
            else:
                fail_reason = meta.get("reason", "未知原因")
                print(f"   [Fail] 算法判定空间极度拥挤，无法拼接: {fail_reason}")

        except Exception as e:
            print(f"   [Error] 算法执行异常: {e}")

    with open(os.path.join(out_dir, "rigid_exhaustive_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n========== 刚体测试完成 ==========")
    print(f"结果已存入 {out_dir}，赶快检查实际几何结构吧！")


if __name__ == "__main__":
    test_exhaustive_with_rigid_body(target_atom_index=13)