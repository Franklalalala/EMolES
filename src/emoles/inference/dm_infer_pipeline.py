import os

from emoles.inference.infer_entry import dptb_infer_from_ase_db, dm_infer_entry


if __name__ == "__main__":
    cwd_ = os.getcwd()
    checkpoint_path =  os.path.join(r'/home/mingkang_nt/hetero_atoms_workspace/1229_test_factory', 'nnenv.ep154.pth')
    output_path = os.path.join(cwd_, 'out_li_clusters')
    db_path = os.path.join(output_path, 'optimized_all.db')
    dptb_infer_from_ase_db(
        ase_db_path=db_path,
        out_path=output_path,
        checkpoint_path=checkpoint_path,
        limit=10,
        device='cuda'
    )
    dm_infer_entry(
        abs_ase_path=db_path,
        results_folder_path=os.path.join(output_path, 'results'),
        gen_esp_cube_flag=True
    )

