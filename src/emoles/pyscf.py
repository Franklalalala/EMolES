import json
import os

import numpy as np
from pyscf import tools, scf
from pyscf.scf.hf import dip_moment


def generate_cube_files(temp_data_file, n_grid, cube_dump_place):
    """Generate cube files for HOMO orbitals and save them in sub-folders named by idx."""

    cwd_ = os.getcwd()
    # Load the saved temporary data
    data = np.load(temp_data_file, allow_pickle=True)
    temp_data = data['temp_data']

    for item in temp_data:
        mol = item['mol']
        outputs = item['outputs']
        tgt_info = item['tgt_info']
        gau_info = item['gau_info']
        idx = item['idx']
        dptb_pred_vs_gau_HOMO_sim = item['dptb_pred_vs_gau_HOMO_sim']

        # Create a sub-folder for each idx inside the cube_dump_place
        sub_folder = os.path.join(cube_dump_place, f'idx_{idx}_sim_{dptb_pred_vs_gau_HOMO_sim:.2g}')
        os.makedirs(sub_folder, exist_ok=True)
        os.chdir(sub_folder)
        if idx < 5:
            tools.cubegen.orbital(mol, 'dptb_predicted_HOMO.cube', outputs['HOMO_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'dptb_label_HOMO.cube', tgt_info['HOMO_coefficients'], nx=n_grid,
                                  ny=n_grid, nz=n_grid)
            tools.cubegen.orbital(mol, 'gau_HOMO.cube', gau_info['HOMO_coefficients'], nx=n_grid, ny=n_grid,
                                  nz=n_grid)
            diff_HOMO = gau_info['HOMO_coefficients'] - outputs['HOMO_coefficients']
            tools.cubegen.orbital(mol, 'gau_prediction_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)
            diff_HOMO = gau_info['HOMO_coefficients'] - tgt_info['HOMO_coefficients']
            tools.cubegen.orbital(mol, 'gau_label_diff_HOMO.cube', diff_HOMO, nx=n_grid, ny=n_grid, nz=n_grid)

        os.chdir(cwd_)


def get_dipole_info(mol, dm):
    mol_dip = dip_moment(mol, dm, unit='DEBYE')
    dip_magnitude = np.linalg.norm(np.array(mol_dip))
    return dip_magnitude
    # dipole_info = {
    #     'Dipole_Moment_Vector_DEBYE': mol_dip.tolist(),
    #     'Dipole_Moment_Norm_DEBYE': float(dip_magnitude),
    # }
    # return dipole_info