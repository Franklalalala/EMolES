import logging
import os
import shutil
from datetime import datetime

from ase.db.core import connect
from ase.io.gaussian import write_gaussian_in
from dpdispatcher import Task

from .utils import (
    build_submission,
    launch_local_job_queue,
    prepare_cooking_dir,
    split_indices,
    wait_for_local_jobs,
)


logging.basicConfig(
    filename="job_monitor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _cleanup_gaussian_outputs(job_folder, remove_chk_flag):
    if remove_chk_flag:
        chk_file = os.path.join(job_folder, "gau.chk")
        if os.path.exists(chk_file):
            os.remove(chk_file)

    for file_name in os.listdir(job_folder):
        if file_name.startswith("core"):
            os.remove(os.path.join(job_folder, file_name))


def _cleanup_gaussian_resubmit_outputs(job_folder, remove_chk_flag):
    if not remove_chk_flag:
        return
    for file_name in os.listdir(job_folder):
        if file_name.endswith("chk") or file_name.startswith("core"):
            os.remove(os.path.join(job_folder, file_name))


def _count_finished_jobs(cooking_path, required_file="gau.log", absent_file="gau.chk"):
    count = 0
    for root, dirs, files in os.walk(cooking_path):
        for dir_name in dirs:
            if dir_name.startswith("id_") or dir_name.startswith("db_seq_id_"):
                folder_path = os.path.join(root, dir_name)
                local_files = os.listdir(folder_path)
                if required_file in local_files and absent_file not in local_files:
                    count += 1
    return count


def _wait_with_progress(cooking_path, total_jobs, job_queue, active_jobs):
    start_time = datetime.now()
    prev_completed_jobs = 0

    def progress_fn():
        nonlocal prev_completed_jobs
        completed_jobs = _count_finished_jobs(cooking_path)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        remaining_jobs = total_jobs - completed_jobs
        if completed_jobs > prev_completed_jobs:
            prev_completed_jobs = completed_jobs
            avg_time_per_job = elapsed_time / completed_jobs
            estimated_remaining_time = avg_time_per_job * remaining_jobs
            logging.info(
                "Completed %s/%s jobs. Estimated remaining time: %.2f hours.",
                completed_jobs,
                total_jobs,
                estimated_remaining_time / 3600,
            )

    wait_for_local_jobs(job_queue, active_jobs, progress_fn=progress_fn, poll_interval=10)


def local_gaussian_from_db(
    n_parallel_jobs,
    n_cpu_per_job,
    db_path,
    gaussian_key_line,
    cmd_line,
    remove_chk_flag=True,
    write_sol_flag=False,
):
    cwd_ = os.getcwd()
    db_path = os.path.abspath(db_path)
    cooking_path = prepare_cooking_dir(reset=False)
    job_folders = []

    with connect(db_path) as db:
        total_jobs = db.count()
        for row in db.select():
            try:
                job_folder = os.path.join(
                    cooking_path,
                    f"id_{row.real_id}_spin_{row.real_spin}_charge_{row.real_charge}",
                )
                real_spin = row.real_spin
                real_charge = row.real_charge
            except Exception:
                job_folder = os.path.join(cooking_path, f"db_seq_id_{row.id - 1}")
                real_spin = 1
                real_charge = 1
            os.makedirs(job_folder, exist_ok=True)
            os.chdir(job_folder)
            an_atoms = row.toatoms()
            with open("gau.gjf", "w") as f_obj:
                write_gaussian_in(
                    fd=f_obj,
                    atoms=an_atoms,
                    properties=[" "],
                    method="",
                    basis=gaussian_key_line,
                    nprocshared=str(n_cpu_per_job),
                    mem="10GB",
                    mult=real_spin,
                    charge=real_charge,
                    chk="gau.chk",
                )
            if write_sol_flag:
                dielectric_constant = row.get("dielectric_constant", 0)
                with open("gau.gjf", "r+") as f_obj:
                    lines = f_obj.readlines()
                    del lines[-1]
                    f_obj.seek(0)
                    f_obj.writelines(lines)
                    f_obj.writelines([f"Eps={dielectric_constant}\n\n\n"])
            job_folders.append(job_folder)

    job_queue, active_jobs = launch_local_job_queue(
        job_folders,
        n_parallel_jobs,
        cmd_line,
        cleanup_fn=lambda folder: _cleanup_gaussian_outputs(folder, remove_chk_flag),
    )
    os.chdir(cwd_)
    _wait_with_progress(cooking_path, total_jobs, job_queue, active_jobs)


def get_file_names(index_file_path):
    with open(index_file_path, "r") as f_obj:
        return [line.strip() for line in f_obj.readlines()]


def local_gaussian_resubmit(
    n_parallel_jobs,
    index_file_path,
    db_src_path,
    common_gau_inp_path,
    cmd_line,
    remove_chk_flag=True,
):
    cwd_ = os.getcwd()
    cooking_path = prepare_cooking_dir(reset=False)
    src_files_list = get_file_names(index_file_path)
    common_gau_inp_path = os.path.abspath(common_gau_inp_path)
    db_src_path = os.path.abspath(db_src_path)
    total_jobs = len(src_files_list)

    job_folders = []
    for fchk_file in src_files_list:
        file_name = os.path.splitext(fchk_file)[0]
        job_folder = os.path.join(cooking_path, file_name)
        os.makedirs(job_folder, exist_ok=True)
        os.chdir(job_folder)
        shutil.copy(src=common_gau_inp_path, dst="gau.gjf")
        shutil.copy(src=os.path.join(db_src_path, fchk_file), dst="old_gau.fchk")
        job_folders.append(job_folder)

    job_queue, active_jobs = launch_local_job_queue(
        job_folders,
        n_parallel_jobs,
        cmd_line,
        cleanup_fn=lambda folder: _cleanup_gaussian_resubmit_outputs(folder, remove_chk_flag),
    )
    os.chdir(cwd_)
    _wait_with_progress(cooking_path, total_jobs, job_queue, active_jobs)


def remote_gaussian_from_db(
    n_parallel_machines,
    main_db_path,
    resrc_info,
    machine_info,
    handler_file_path,
):
    cwd_ = os.getcwd()
    abs_handler_file_path = os.path.abspath(handler_file_path)
    handler_file_name = os.path.basename(handler_file_path)
    cooking_path = prepare_cooking_dir(reset=True)
    task_list = []

    with connect(main_db_path) as main_db:
        all_rows = list(main_db.select())
        total_n_mols = len(all_rows)
        for shard_id, shard_indices in enumerate(split_indices(total_n_mols, n_parallel_machines, shuffle=True)):
            os.chdir(cooking_path)
            os.makedirs(str(shard_id))
            os.chdir(str(shard_id))
            shutil.copy(src=abs_handler_file_path, dst=handler_file_name)
            with connect("raw.db") as dump_db:
                for row_idx in shard_indices:
                    dump_db.write(all_rows[row_idx])
            task_list.append(
                Task(
                    command=fr"python {handler_file_name} 2>&1 ",
                    task_work_path=f"{shard_id}/",
                    forward_files=[f"{cooking_path}/{shard_id}/*"],
                    backward_files=["cooking"],
                )
            )
    os.chdir(cwd_)
    build_submission(cooking_path, machine_info, resrc_info, task_list)


def remote_gaussian_resubmit(
    n_parallel_machines,
    resrc_info,
    machine_info,
    db_src_path,
    common_file_list,
    handler_file_name,
):
    cwd_ = os.getcwd()
    cooking_path = prepare_cooking_dir(reset=True)
    task_list = []

    sorted_db_src_files = sorted(os.listdir(db_src_path))
    total_n_mols = len(sorted_db_src_files)
    for shard_id, shard_indices in enumerate(split_indices(total_n_mols, n_parallel_machines, shuffle=True)):
        os.chdir(cooking_path)
        os.makedirs(str(shard_id))
        os.chdir(str(shard_id))
        for common_file in common_file_list:
            shutil.copy(src=os.path.join(cwd_, common_file), dst=common_file)
        with open("file_names", "w") as f_obj:
            for real_idx in shard_indices:
                f_obj.write(sorted_db_src_files[real_idx])
                f_obj.write("\n")
        task_list.append(
            Task(
                command=fr"python {handler_file_name} 2>&1 ",
                task_work_path=f"{shard_id}/",
                forward_files=[f"{cooking_path}/{shard_id}/*"],
                backward_files=["cooking"],
            )
        )

    os.chdir(cwd_)
    build_submission(cooking_path, machine_info, resrc_info, task_list)
