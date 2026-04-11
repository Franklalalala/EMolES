import json
import multiprocessing as mp
import os
import subprocess
import traceback


def resolve_gpu_ids(device="cuda", gpus=None):
    dev = str(device).lower().strip()
    if dev == "cpu":
        return []
    if gpus:
        return [int(x) for x in gpus]
    if dev.startswith("cuda:"):
        return [int(dev.split(":")[1])]
    try:
        import torch

        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0
    if n_gpus <= 0:
        raise RuntimeError("No CUDA GPU detected, but device is not cpu.")
    return list(range(n_gpus))


def build_worker_gpu_plan(
    device="cuda",
    gpus=None,
    workers_per_gpu=1,
    cpu_workers=None,
    n_tasks=None,
):
    dev = str(device).lower().strip()
    if dev == "cpu":
        plan = [None] * max(1, int(cpu_workers or n_tasks or 1))
    else:
        if int(workers_per_gpu) <= 0:
            raise ValueError("workers_per_gpu must be >= 1")
        plan = []
        for gpu_id in resolve_gpu_ids(device=device, gpus=gpus):
            plan.extend([int(gpu_id)] * int(workers_per_gpu))
    if n_tasks is not None:
        return plan[: int(n_tasks)]
    return plan


def set_thread_env(cpu_threads_per_worker=None):
    if cpu_threads_per_worker is None or int(cpu_threads_per_worker) <= 0:
        return
    value = str(int(cpu_threads_per_worker))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[key] = value


def configure_worker_env(gpu_id=None, cpu_threads_per_worker=None):
    set_thread_env(cpu_threads_per_worker)
    if gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

    try:
        import torch

        if cpu_threads_per_worker is not None and int(cpu_threads_per_worker) > 0:
            try:
                torch.set_num_threads(int(cpu_threads_per_worker))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                precision = os.environ.get("EMOLES_FLOAT32_MATMUL_PRECISION", "highest")
                torch.set_float32_matmul_precision(precision)
                allow_tf32 = str(precision).lower() != "highest"
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = allow_tf32
            except Exception:
                pass
        if gpu_id is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
    except Exception:
        pass


def query_gpu_mem_mb(gpu_id):
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used // (1024 * 1024)), int(info.total // (1024 * 1024))
    except Exception:
        pass

    try:
        cmd = [
            "nvidia-smi",
            "-i",
            str(int(gpu_id)),
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        parts = [part.strip() for part in output.split(",")]
        if len(parts) >= 2:
            return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None
    return None


def write_json_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_obj:
        json.dump(payload, f_obj, indent=2, ensure_ascii=False)


def _worker_entry(worker_main, worker_spec, cpu_threads_per_worker, result_queue):
    try:
        configure_worker_env(
            gpu_id=worker_spec.get("gpu_id"),
            cpu_threads_per_worker=cpu_threads_per_worker,
        )
        worker_main(worker_spec)
        result_queue.put(
            {
                "worker_id": int(worker_spec["worker_id"]),
                "gpu_id": worker_spec.get("gpu_id"),
                "status": "ok",
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": int(worker_spec["worker_id"]),
                "gpu_id": worker_spec.get("gpu_id"),
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_worker_pool(
    worker_specs,
    worker_main,
    cpu_threads_per_worker=None,
    poll_interval_sec=5.0,
):
    if not worker_specs:
        return []

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    recorded_worker_ids = set()

    for worker_spec in worker_specs:
        proc = ctx.Process(
            target=_worker_entry,
            args=(worker_main, worker_spec, cpu_threads_per_worker, result_queue),
        )
        proc.start()
        processes.append((worker_spec, proc))

    results = []
    while True:
        try:
            msg = result_queue.get(timeout=float(poll_interval_sec))
            results.append(msg)
            recorded_worker_ids.add(int(msg["worker_id"]))
        except Exception:
            pass

        all_exited = all((not proc.is_alive()) for _, proc in processes)
        if all_exited and len(recorded_worker_ids) >= len(processes):
            break
        if all_exited:
            break

    for worker_spec, proc in processes:
        proc.join()
        if proc.exitcode not in (0, None):
            already_recorded = any(
                result["worker_id"] == worker_spec["worker_id"] for result in results
            )
            if not already_recorded:
                results.append(
                    {
                        "worker_id": worker_spec["worker_id"],
                        "gpu_id": worker_spec.get("gpu_id"),
                        "status": "error",
                        "error": f"worker exited with code {proc.exitcode}",
                    }
                )

    return results
