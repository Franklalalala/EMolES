# using dpdispatcher to perform multiple xtb calculations in a single node
import math
import os
import shutil

import numpy as np
from ase.db.core import connect
from ase.io import write, read
from dpdispatcher import Task, Submission, Machine, Resources


def local_xtb(n_parallel_job, n_cpu_per_job, db_path, cmd_line_head, **kwargs):
    cwd_ = os.getcwd()
    db_path = os.path.abspath(db_path)
    # prepare
    cooking_path = os.path.abspath('cooking')
    os.makedirs(cooking_path)
    mach_para = {
        'batch_type': "Shell",
        'context_type': "LazyLocalContext",
        'remote_root': '/root/test_dpdispatcher',
        'remote_profile': {},
        'local_root': cooking_path,
        'retry_count': 2
    }

    resrc_para = {
        'number_node': 1,
        'cpu_per_node': n_cpu_per_job,
        'gpu_per_node': 0,
        'group_size': n_parallel_job,
        'queue_name': "LBG_CPU",
        'envs': {
            "OMP_STACKSIZE": "4G",
            "OMP_NUM_THREADS": "3,1",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "MKL_NUM_THREADS": "3"
        },
        'strategy': {
            'ratio_unfinished': 0.1
        }
    }

    task_list = []
    with connect(db_path) as db:
        for a_row in db.select():
            os.chdir(cooking_path)
            id_name = f'id_{a_row.real_id}_spin_{a_row.real_spin}_charge_{a_row.real_charge}'
            os.makedirs(id_name)
            os.chdir(id_name)
            an_atoms = a_row.toatoms()
            write('raw.xyz', an_atoms)

            with open('.CHRG', 'w') as f:
                f.write(f'{a_row.real_charge}')
            with open('.UHF', 'w') as f:
                f.write(f'{a_row.real_spin - 1}')

            if cmd_line_head == 'xtb':
                a_task = Task(
                    command=' '.join([f'{cmd_line_head} raw.xyz >> output.txt']),
                    task_work_path=f'{id_name}/',
                    forward_files=[f'{cooking_path}/{id_name}/*'],
                    backward_files=['output.xyz']
                )
            else:
                a_task = Task(
                    command=' '.join([f'xtb --opt tight raw.xyz >> output.txt']),
                    task_work_path=f'{id_name}/',
                    forward_files=[f'{cooking_path}/{id_name}/*'],
                    backward_files=['xtbopt.xyz', 'output.xyz']
                )
            task_list.append(a_task)

    submission = Submission(
        work_base=cooking_path,
        machine=Machine.load_from_dict(machine_dict=mach_para),
        resources=Resources.load_from_dict(resources_dict=resrc_para),
        task_list=task_list,
    )
    submission.run_submission()


def remote_xtb(n_parallel_machines, main_db_path, resrc_info, machine_info, handler_file_path, handler_inputs: dict):
    cwd_ = os.getcwd()
    # prepare
    abs_handler_file_path = os.path.abspath(handler_file_path)
    handler_file_name = os.path.basename(handler_file_path)
    cooking_path = os.path.abspath('cooking')
    if os.path.exists(cooking_path):
        print('Found previous workbase. It will be cleared.')
        shutil.rmtree(cooking_path)
    os.makedirs(cooking_path)
    task_list = []
    with connect(main_db_path) as main_db:
        total_n_mols = main_db.count()
        sub_n_mols = math.ceil(total_n_mols / n_parallel_machines)
        for i in range(n_parallel_machines):
            os.chdir(cooking_path)
            os.makedirs(f'{str(i)}')
            os.chdir(f'{str(i)}')
            shutil.copy(src=abs_handler_file_path, dst=handler_file_name)
            if i < n_parallel_machines-1:
                with connect('raw.db') as dump_db:
                    for row_idx, a_row in enumerate(main_db.select()):
                        if row_idx in np.arange(start=i * sub_n_mols, stop=(i + 1) * sub_n_mols):
                            dump_db.write(a_row)
            else:
                with connect('raw.db') as dump_db:
                    for row_idx, a_row in enumerate(main_db.select()):
                        if row_idx >= i * sub_n_mols:
                            dump_db.write(a_row)
            # task
            n_parallel_job = handler_inputs['n_parallel_job']
            n_cpu_per_job = handler_inputs['n_cpu_per_job']
            cmd_line_head = handler_inputs['cmd_line_head']
            a_task = Task(
                command=fr'python {handler_file_name} --n_parallel_job {n_parallel_job} --cmd_line_head {cmd_line_head} --n_cpu_per_job {n_cpu_per_job} 2>&1 ',
                task_work_path=f'{str(i)}/',
                forward_files=[f'{cooking_path}/{str(i)}/*'],
                backward_files=[f'cooking']
            )
            task_list.append(a_task)
    os.chdir(cwd_)
    # submission
    submission = Submission(
        work_base=cooking_path,
        machine=Machine.load_from_dict(machine_dict=machine_info),
        resources=Resources.load_from_dict(resources_dict=resrc_info),
        task_list=task_list,
    )
    submission.run_submission()



