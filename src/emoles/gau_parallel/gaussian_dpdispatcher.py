import math
import os
import random
import shutil
import subprocess
import time
import logging
from queue import Queue
from threading import Thread

import numpy as np
from ase.db.core import connect
from ase.io.gaussian import read_gaussian_out, write_gaussian_in
from dpdispatcher import Task, Submission, Machine, Resources

from datetime import datetime

# Configure logging
logging.basicConfig(filename='job_monitor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def submit_job(job_folder, cmd_line):
    os.chdir(job_folder)
    process = subprocess.Popen(cmd_line.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def monitor_jobs(job_queue, active_jobs, cmd_line, remove_chk_flag):
    while True:
        for job_folder, process in list(active_jobs.items()):
            retcode = process.poll()
            if retcode is not None:  # Job finished
                del active_jobs[job_folder]
                # Remove unwanted output file
                chk_file = os.path.join(job_folder, 'gau.chk')
                if remove_chk_flag and os.path.exists(chk_file):
                    os.remove(chk_file)

                # Remove corrupted core files
                for file in os.listdir(job_folder):
                    if file.startswith('core'):
                        os.remove(os.path.join(job_folder, file))

                # Submit a new job if there are any left
                if not job_queue.empty():
                    next_job = job_queue.get()
                    active_jobs[next_job] = submit_job(next_job, cmd_line)
        time.sleep(1)  # Check every second


def chk_finished_jobs(cooking_path):
    count = 0
    for root, dirs, files in os.walk(cooking_path):
        for dir_name in dirs:
            if dir_name.startswith('id_'):
                folder_path = os.path.join(root, dir_name)
                if 'gau.log' in os.listdir(folder_path):
                    if 'gau.chk' not in os.listdir(folder_path):
                        count += 1
    return count


def local_gaussian(n_parallel_jobs, n_cpu_per_job, db_path, gaussian_key_line, cmd_line, remove_chk_flag=True, write_sol_flag=False):
    cwd_ = os.getcwd()
    db_path = os.path.abspath(db_path)
    cooking_path = os.path.abspath('cooking')
    os.makedirs(cooking_path, exist_ok=True)

    job_queue = Queue()
    with connect(db_path) as db:
        total_jobs = db.count()
        for a_row in db.select():
            try:
                job_folder = os.path.join(cooking_path,
                                          f'id_{a_row.real_id}_spin_{a_row.real_spin}_charge_{a_row.real_charge}')
                real_spin = a_row.real_spin
                real_charge = a_row.real_charge
            except:
                job_folder = os.path.join(cooking_path, f'db_seq_id_{a_row.id-1}')
                real_spin = 1
                real_charge = 1
            os.makedirs(job_folder, exist_ok=True)
            os.chdir(job_folder)
            an_atoms = a_row.toatoms()
            with open(file='gau.gjf', mode='w') as f:
                write_gaussian_in(fd=f,
                                  atoms=an_atoms,
                                  properties=[' '],
                                  method='',
                                  basis=gaussian_key_line,
                                  nprocshared=str(n_cpu_per_job),
                                  mem='10GB',
                                  mult=real_spin,
                                  charge=real_charge,
                                  chk='gau.chk',
                                  )
            if write_sol_flag:
                with open('gau.gjf', "r+") as file:
                    lines = file.readlines()
                    del lines[-1]
                    file.seek(0)
                    file.writelines(lines)
                    file.writelines(['Eps=18.5\n',
                                     'EpsInf=1.415\n',
                                     'HbondAcidity=0\n',
                                     'HbondBasicity=0.735\n',
                                     'SurfaceTensionAtInterface=20.2\n',
                                     'CarbonAromaticity=0\n',
                                     'ElectronegativeHalogenicity=0\n\n\n',
                                     ])
            job_queue.put(job_folder)

    # Dictionary to keep track of active jobs
    active_jobs = {}

    # Submit initial jobs
    for _ in range(min(n_parallel_jobs, job_queue.qsize())):
        job_folder = job_queue.get()
        active_jobs[job_folder] = submit_job(job_folder, cmd_line)

    # Start monitoring thread
    monitor_thread = Thread(target=monitor_jobs, args=(job_queue, active_jobs, cmd_line, remove_chk_flag), daemon=True)
    monitor_thread.start()

    start_time = datetime.now()
    prev_completed_jobs = 0
    os.chdir(cwd_)
    # Keep the script running
    while not job_queue.empty() or active_jobs:
        time.sleep(10)
        completed_jobs = chk_finished_jobs(cooking_path)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        remaining_jobs = total_jobs - completed_jobs
        if completed_jobs > prev_completed_jobs:
            prev_completed_jobs = completed_jobs
            avg_time_per_job = elapsed_time / completed_jobs
            estimated_remaining_time = avg_time_per_job * remaining_jobs
            logging.info(f"Completed {completed_jobs}/{total_jobs} jobs. Estimated remaining time: {estimated_remaining_time / 3600:.2f} hours.")


def remote_gaussian(n_parallel_machines, main_db_path, resrc_info, machine_info, handler_file_path):
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
        random_idx_list = list(range(total_n_mols))
        random.shuffle(random_idx_list)
        if total_n_mols % n_parallel_machines == 0:
            sub_n_mols = total_n_mols // n_parallel_machines
            actual_machine_used = n_parallel_machines
        else:
            sub_n_mols = math.ceil(total_n_mols / n_parallel_machines)
            actual_machine_used = total_n_mols // sub_n_mols + 1
        for i in range(actual_machine_used):
            os.chdir(cooking_path)
            os.makedirs(f'{str(i)}')
            os.chdir(f'{str(i)}')
            shutil.copy(src=abs_handler_file_path, dst=handler_file_name)
            if i < actual_machine_used - 1:
                with connect('raw.db') as dump_db:
                    for row_idx, a_row in enumerate(main_db.select()):
                        if row_idx in random_idx_list[i * sub_n_mols: (i + 1) * sub_n_mols]:
                            dump_db.write(a_row)
            else:
                with connect('raw.db') as dump_db:
                    for row_idx, a_row in enumerate(main_db.select()):
                        if row_idx in random_idx_list[i * sub_n_mols: ]:
                            dump_db.write(a_row)
            # task
            a_task = Task(
                command=fr'python {handler_file_name} 2>&1 ',
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


