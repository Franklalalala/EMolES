import os

from emoles.inference.infer_entry import dptb_infer_from_ase_db
from emoles.inference.common_tools import smile_2_db, generate_cube_files
from emoles.py3Dmol import cubes_2_htmls


smile_path = r'filtered_cho_smiles.txt'
ase_db_path='dump.db'
# # 0. transfer smiles to db
smile_2_db(
    smile_path=smile_path,
    db_path=ase_db_path,
    fail_smile_path='failed_smiles.txt',
    maxAttempts=10000 # how many times try to assign 3D croods to a smiles
)


# 1. get predicted hamiltonian matrix
input_json_path = os.path.abspath(r'/root/emoles_workspace/test_old_api/train_config.json')
pth_path = os.path.abspath(r'/root/emoles_workspace/test_old_api/nnenv.best.pth')
device='cuda'
out_path='ham_output'
limit = 3

dptb_infer_from_ase_db(ase_db_path=ase_db_path, out_path=out_path, limit=limit,
                       input_path=input_json_path, checkpoint_path=pth_path,
                       device=device)

# 2. get cube files with pyscf overlap
generate_cube_files(ase_db_path=ase_db_path, out_path=out_path, n_grid=75, dm_flag=True, limit=limit, basis='6311gdp')

# 3. visualize cube files
cubes_2_htmls(out_path='ham_output', iso_value=0.03)



