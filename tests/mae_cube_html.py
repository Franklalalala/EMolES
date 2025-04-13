import os
import json
from pprint import pprint
import shutil
from emoles.loss import test_with_npy
from emoles.pyscf import generate_cube_files
from emoles.py3Dmol import cubes_2_htmls


if __name__ == '__main__':
    abs_ase_path = r'dump.db'
    npy_folder_path = r'output'
    gau_npy_folder_path = r'/share/lmk_1399/1104_no_li_workbase/matrix_dump_split/6311gdp/test'
    n_grid = 75
    convention = '6311gdp'

    # Create a temporary folder for storing intermediate results
    os.makedirs(npy_folder_path, exist_ok=True)
    temp_data_file = os.path.abspath('temp_data.npz')
    cube_dump_place = os.path.abspath('cubes')
    if os.path.exists(cube_dump_place):
        shutil.rmtree(cube_dump_place)
    if os.path.exists(temp_data_file):
        os.remove(temp_data_file)
    os.makedirs(cube_dump_place)

    # Call the test_with_npy function (assuming this function is defined elsewhere)
    total_error_dict = test_with_npy(abs_ase_path=abs_ase_path, npy_folder_path=npy_folder_path,
                                     temp_data_file=temp_data_file, gau_npy_folder_path=gau_npy_folder_path,
                                     united_overlap_flag=True, convention=convention, mol_charge=0)

    # Print results
    pprint(total_error_dict)
    with open('test_results.json', 'w') as f:
        json.dump(total_error_dict, f, indent=2)

    # Generate cube files after test_with_npy
    generate_cube_files(temp_data_file, n_grid, cube_dump_place)

    # Assuming batch_visualize_cube_file is defined elsewhere
    cubes_2_htmls(cube_dump_place, 0.03)
