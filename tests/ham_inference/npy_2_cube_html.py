from emoles.py3Dmol import cubes_2_htmls
from emoles.inference.infer_entry import get_ham_info_from_npy


if __name__ == '__main__':
    abs_ase_path = r'/home/mingkang_nt/dptb_loss_workspace/cluster/16_slow_slow/test_1213/dump.db'
    npy_folder_path = r'/home/mingkang_nt/dptb_loss_workspace/cluster/16_slow_slow/infer_test/output/npy'
    # gau_npy_folder_path = r'/share/lmk_1399/1104_no_li_workbase/matrix_dump_split/6311gdp/test'
    n_grid = 75
    convention = 'def2svp'
    get_ham_info_from_npy(
        ase_db_path=abs_ase_path,
        npy_folder_path=npy_folder_path,
        convention=convention,
        max_items=5,
        max_cube_save=3, # how many in max_items to gen cubes
        cube_grid=n_grid
    )
    # Assuming batch_visualize_cube_file is defined elsewhere
    cubes_2_htmls(npy_folder_path, 0.03)
