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
    process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return process


def monitor_jobs(job_queue, active_jobs, cmd_line, remove_chk_flag):
    while True:
        for job_folder, process in list(active_jobs.items()):
            retcode = process.poll()
            if retcode is not None:  # Job finished
                del active_jobs[job_folder]
                # Remove unwanted output file
                if remove_chk_flag:
                    for a_file in os.listdir(job_folder):
                        if a_file.endswith('chk') or a_file.startswith('core'):
                            os.remove(os.path.join(job_folder, a_file))

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


def get_file_names(index_file_path):
    file_names_list = []
    with open(index_file_path, 'r') as f:
        for a_line in f.readlines():
            a_line = a_line.strip()
            file_names_list.append(a_line)
    return file_names_list


def local_gaussian(n_parallel_jobs, index_file_path, db_src_path, common_gau_inp_path, cmd_line, remove_chk_flag=True):
    cwd_ = os.getcwd()
    cooking_path = os.path.abspath('cooking')
    os.makedirs(cooking_path, exist_ok=True)
    src_files_list = get_file_names(index_file_path)
    common_gau_inp_path = os.path.abspath(common_gau_inp_path)
    db_src_path = os.path.abspath(db_src_path)
    total_jobs = len(src_files_list)

    job_queue = Queue()
    for a_fchk_file in src_files_list:
        file_name = os.path.splitext(a_fchk_file)[0]
        job_folder = os.path.join(cooking_path, file_name)
        os.makedirs(job_folder, exist_ok=True)
        os.chdir(job_folder)
        shutil.copy(src=common_gau_inp_path, dst='gau.gjf')
        shutil.copy(src=os.path.join(db_src_path, a_fchk_file), dst='old_gau.fchk')
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


def remote_gaussian(n_parallel_machines, resrc_info, machine_info, db_src_path, common_file_list, handler_file_name):
    cwd_ = os.getcwd()
    # prepare
    cooking_path = os.path.abspath('cooking')
    if os.path.exists(cooking_path):
        print('Found previous workbase. It will be cleared.')
        shutil.rmtree(cooking_path)
    os.makedirs(cooking_path)
    task_list = []

    sorted_db_src_files = sorted(os.listdir(db_src_path))
    total_n_mols = len(sorted_db_src_files)
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
        for a_common_file in common_file_list:
            shutil.copy(src=os.path.join(cwd_, a_common_file), dst=a_common_file)
        if i < actual_machine_used - 1:
            with open('file_names', 'w') as f:
                for real_idx in random_idx_list[i * sub_n_mols: (i + 1) * sub_n_mols]:
                    a_fchk_file = sorted_db_src_files[real_idx]
                    f.write(a_fchk_file)
                    f.write('\n')
        else:
            with open('file_names', 'w') as f:
                for real_idx in random_idx_list[i * sub_n_mols: ]:
                    a_fchk_file = sorted_db_src_files[real_idx]
                    f.write(a_fchk_file)
                    f.write('\n')
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


