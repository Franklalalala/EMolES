import os
import shutil

from ase.db.core import connect
from ase.io import write
from dpdispatcher import Task

from .utils import build_submission, prepare_cooking_dir, split_indices


def local_xtb(n_parallel_job, n_cpu_per_job, db_path, cmd_line_head, **kwargs):
    cwd_ = os.getcwd()
    db_path = os.path.abspath(db_path)
    cooking_path = prepare_cooking_dir(reset=False)
    mach_para = {
        "batch_type": "Shell",
        "context_type": "LazyLocalContext",
        "remote_root": "/root/test_dpdispatcher",
        "remote_profile": {},
        "local_root": cooking_path,
        "retry_count": 2,
    }
    resrc_para = {
        "number_node": 1,
        "cpu_per_node": n_cpu_per_job,
        "gpu_per_node": 0,
        "group_size": n_parallel_job,
        "queue_name": "LBG_CPU",
        "envs": {
            "OMP_STACKSIZE": "4G",
            "OMP_NUM_THREADS": "3,1",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "MKL_NUM_THREADS": "3",
        },
        "strategy": {"ratio_unfinished": 0.1},
    }

    task_list = []
    with connect(db_path) as db:
        for row in db.select():
            os.chdir(cooking_path)
            id_name = f"id_{row.real_id}_spin_{row.real_spin}_charge_{row.real_charge}"
            os.makedirs(id_name)
            os.chdir(id_name)
            an_atoms = row.toatoms()
            write("raw.xyz", an_atoms)

            with open(".CHRG", "w") as f_obj:
                f_obj.write(f"{row.real_charge}")
            with open(".UHF", "w") as f_obj:
                f_obj.write(f"{row.real_spin - 1}")

            if cmd_line_head == "xtb":
                command = f"{cmd_line_head} raw.xyz >> output.txt"
                backward_files = ["output.xyz"]
            else:
                command = "xtb --opt tight raw.xyz >> output.txt"
                backward_files = ["xtbopt.xyz", "output.xyz"]

            task_list.append(
                Task(
                    command=command,
                    task_work_path=f"{id_name}/",
                    forward_files=[f"{cooking_path}/{id_name}/*"],
                    backward_files=backward_files,
                )
            )

    os.chdir(cwd_)
    build_submission(cooking_path, mach_para, resrc_para, task_list)


def remote_xtb(
    n_parallel_machines,
    main_db_path,
    resrc_info,
    machine_info,
    handler_file_path,
    handler_inputs,
):
    cwd_ = os.getcwd()
    abs_handler_file_path = os.path.abspath(handler_file_path)
    handler_file_name = os.path.basename(handler_file_path)
    cooking_path = prepare_cooking_dir(reset=True)
    task_list = []

    with connect(main_db_path) as main_db:
        all_rows = list(main_db.select())
        total_n_mols = len(all_rows)
        for shard_id, shard_indices in enumerate(split_indices(total_n_mols, n_parallel_machines, shuffle=False)):
            os.chdir(cooking_path)
            os.makedirs(str(shard_id))
            os.chdir(str(shard_id))
            shutil.copy(src=abs_handler_file_path, dst=handler_file_name)
            with connect("raw.db") as dump_db:
                for row_idx in shard_indices:
                    dump_db.write(all_rows[row_idx])

            n_parallel_job = handler_inputs["n_parallel_job"]
            n_cpu_per_job = handler_inputs["n_cpu_per_job"]
            cmd_line_head = handler_inputs["cmd_line_head"]
            task_list.append(
                Task(
                    command=(
                        f"python {handler_file_name} --n_parallel_job {n_parallel_job} "
                        f"--cmd_line_head {cmd_line_head} --n_cpu_per_job {n_cpu_per_job} 2>&1 "
                    ),
                    task_work_path=f"{shard_id}/",
                    forward_files=[f"{cooking_path}/{shard_id}/*"],
                    backward_files=["cooking"],
                )
            )
    os.chdir(cwd_)
    build_submission(cooking_path, machine_info, resrc_info, task_list)
