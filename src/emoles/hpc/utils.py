import inspect
import math
import os
import random
import shutil
import subprocess
import time
from queue import Queue
from threading import Thread

from dpdispatcher import Machine, Resources, Submission


_BOHRIUM_BATCH_TYPES = {"bohrium", "dpcloudserver", "lebesgue"}
_BOHRIUM_CONTEXT_TYPES = {"bohriumcontext", "dpcloudservercontext", "lebesguecontext"}


def prepare_cooking_dir(dirname="cooking", reset=False):
    cooking_path = os.path.abspath(dirname)
    if reset and os.path.exists(cooking_path):
        print("Found previous workbase. It will be cleared.")
        shutil.rmtree(cooking_path)
    os.makedirs(cooking_path, exist_ok=True)
    return cooking_path


def split_indices(total_items, n_groups, shuffle=False):
    indices = list(range(total_items))
    if shuffle:
        random.shuffle(indices)
    if total_items == 0:
        return []
    chunk_size = math.ceil(total_items / n_groups)
    return [indices[i : i + chunk_size] for i in range(0, total_items, chunk_size)]


def submit_local_job(job_folder, cmd_line):
    os.chdir(job_folder)
    return subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def monitor_local_jobs(job_queue, active_jobs, cmd_line, cleanup_fn=None, poll_interval=1):
    while True:
        for job_folder, process in list(active_jobs.items()):
            retcode = process.poll()
            if retcode is None:
                continue

            del active_jobs[job_folder]
            if cleanup_fn is not None:
                cleanup_fn(job_folder)

            if not job_queue.empty():
                next_job = job_queue.get()
                active_jobs[next_job] = submit_local_job(next_job, cmd_line)
        time.sleep(poll_interval)


def launch_local_job_queue(job_folders, n_parallel_jobs, cmd_line, cleanup_fn=None):
    job_queue = Queue()
    for job_folder in job_folders:
        job_queue.put(job_folder)

    active_jobs = {}
    for _ in range(min(n_parallel_jobs, job_queue.qsize())):
        job_folder = job_queue.get()
        active_jobs[job_folder] = submit_local_job(job_folder, cmd_line)

    monitor_thread = Thread(
        target=monitor_local_jobs,
        args=(job_queue, active_jobs, cmd_line, cleanup_fn),
        daemon=True,
    )
    monitor_thread.start()
    return job_queue, active_jobs


def wait_for_local_jobs(job_queue, active_jobs, progress_fn=None, poll_interval=10):
    while not job_queue.empty() or active_jobs:
        time.sleep(poll_interval)
        if progress_fn is not None:
            progress_fn()


def _patch_dargs_allow_ref():
    try:
        from dargs import Argument
    except ImportError:
        return

    for method_name in ("normalize_value", "check_value"):
        original = getattr(Argument, method_name, None)
        if original is None:
            continue
        signature = inspect.signature(original)
        if "allow_ref" in signature.parameters:
            continue

        def _wrapped(self, *args, allow_ref=False, _original=original, **kwargs):
            return _original(self, *args, **kwargs)

        setattr(Argument, method_name, _wrapped)


def _is_bohrium_machine(machine_info):
    batch_type = str(machine_info.get("batch_type", "")).strip().lower()
    context_type = str(machine_info.get("context_type", "")).strip().lower()
    return batch_type in _BOHRIUM_BATCH_TYPES or context_type in _BOHRIUM_CONTEXT_TYPES


def _normalize_bohrium_remote_profile(remote_profile):
    normalized = dict(remote_profile or {})

    project_id = normalized.get("project_id", normalized.get("program_id"))
    if project_id not in (None, ""):
        project_id = int(project_id)
        normalized["project_id"] = project_id

    input_data = dict(normalized.get("input_data") or {})
    for key in ("job_type", "log_file", "job_name", "scass_type", "platform", "image_name"):
        if input_data.get(key) not in (None, ""):
            input_data[key] = str(input_data[key]).strip()
    if input_data.get("disk_size") not in (None, ""):
        input_data["disk_size"] = int(input_data["disk_size"])
    normalized["input_data"] = input_data
    return normalized


def _build_machine(machine_info):
    _patch_dargs_allow_ref()
    if not _is_bohrium_machine(machine_info):
        return Machine.load_from_dict(machine_dict=machine_info)

    from dpdispatcher.contexts.dp_cloud_server_context import BohriumContext
    from dpdispatcher.machines.dp_cloud_server import Bohrium

    context = BohriumContext(
        local_root=machine_info["local_root"],
        remote_root=machine_info.get("remote_root"),
        remote_profile=_normalize_bohrium_remote_profile(machine_info.get("remote_profile", {})),
    )
    machine_kwargs = {
        key: value
        for key, value in machine_info.items()
        if key not in {"batch_type", "context_type", "local_root", "remote_root", "remote_profile"}
    }
    return Bohrium(context=context, **machine_kwargs)


def _build_resources(resrc_info):
    resources_dict = dict(resrc_info)
    _patch_dargs_allow_ref()

    load_from_dict = getattr(Resources, "load_from_dict", None)
    if callable(load_from_dict):
        try:
            return load_from_dict(resources_dict=resources_dict)
        except TypeError as exc:
            if "allow_ref" not in str(exc):
                raise

    return Resources(**resources_dict)


def build_submission(cooking_path, machine_info, resrc_info, task_list, run_submission_kwargs=None):
    machine_info = dict(machine_info)
    context_type = str(machine_info.get("context_type", "")).strip()
    if context_type == "LocalContext":
        run_tag = f"run_{int(time.time() * 1000)}"
        local_root = os.path.abspath(
            machine_info.get("local_root") or os.path.join(cooking_path, "local_root")
        )
        remote_root = os.path.abspath(
            machine_info.get("remote_root") or os.path.join(cooking_path, "remote_root")
        )
        if local_root == remote_root:
            local_root = os.path.join(local_root, "local")
            remote_root = os.path.join(remote_root, "remote")
        local_root = os.path.join(local_root, run_tag)
        remote_root = os.path.join(remote_root, run_tag)
        os.makedirs(local_root, exist_ok=True)
        os.makedirs(remote_root, exist_ok=True)
        machine_info["local_root"] = local_root
        machine_info["remote_root"] = remote_root

    submission = Submission(
        work_base=cooking_path,
        machine=_build_machine(machine_info),
        resources=_build_resources(resrc_info),
        task_list=task_list,
    )
    run_submission_kwargs = dict(run_submission_kwargs or {})
    try:
        return submission.run_submission(**run_submission_kwargs)
    except TypeError as exc:
        if run_submission_kwargs and "unexpected keyword argument" in str(exc):
            return submission.run_submission()
        raise
