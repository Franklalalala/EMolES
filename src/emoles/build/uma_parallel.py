import argparse
import multiprocessing as mp
import os
import shutil
from typing import List, Optional

from ase.db import connect

from emoles.build.uma_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_NAME,
    DEFAULT_WORKSPACE,
    prepare_input_source,
)
from emoles.build.uma_parallel_utils import (
    _build_worker_gpu_plan,
    _merge_worker_dbs,
    _prepare_model_assets,
    _prepare_workspace,
    _safe_log,
    _set_thread_env,
    _worker_loop,
)
from emoles.utils.parallel import run_ase_db_task_queue_pool


def _spawn_uma_worker(
    *,
    slot_id,
    attempt,
    gpu_id,
    ctx,
    task_queue,
    result_queue,
    stop_event,
    active_input_db,
    checkpoint_path,
    model_name,
    workspace,
    fmax,
    max_steps,
    cpu_threads_per_worker,
    shard_db_dir,
    empty_cache_every,
):
    worker_db_path = os.path.join(shard_db_dir, f"slot_{slot_id:03d}__a{attempt:04d}.db")
    proc = ctx.Process(
        target=_worker_loop,
        args=(
            slot_id,
            attempt,
            gpu_id,
            active_input_db,
            worker_db_path,
            checkpoint_path,
            model_name,
            workspace,
            fmax,
            max_steps,
            cpu_threads_per_worker,
            task_queue,
            result_queue,
            stop_event,
            int(empty_cache_every),
        ),
    )
    proc.start()
    return proc, worker_db_path


def entry(
    input_db: str = None,
    smiles: list = None,
    workspace: str = DEFAULT_WORKSPACE,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cuda",
    gpus: Optional[List[str]] = None,
    workers_per_gpu: int = 4,
    cpu_workers: Optional[int] = None,
    cpu_threads_per_worker: Optional[int] = None,
    fmax: float = 0.05,
    max_steps: int = 200,
    verbose: bool = False,
    show_progress: bool = True,
    queue_size: Optional[int] = None,
    write_xyz: bool = False,
    keep_worker_dbs: bool = False,
    warmup_workers_per_gpu: int = 2,
    ramp_step_per_gpu: int = 1,
    ramp_interval_sec: float = 60.0,
    stable_window_sec: float = 60.0,
    gpu_headroom_gb: float = 13.0,
    init_timeout_sec: float = 900.0,
    startup_stagger_sec: float = 0.2,
    global_init_concurrency: int = 1,
    init_concurrency_per_gpu: int = 1,
    max_restarts_per_slot: int = 200,
    restart_backoff_sec: float = 2.0,
    empty_cache_every: int = 0,
    pytorch_alloc_conf: Optional[str] = None,
    task_timeout_sec: Optional[float] = None,
) -> str:
    os.makedirs(workspace, exist_ok=True)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if pytorch_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_alloc_conf)

    out_xyz_dir, shard_db_dir, out_db_path, fail_log_path = _prepare_workspace(workspace, write_xyz=write_xyz)
    active_input_db = prepare_input_source(workspace, input_db, smiles)
    if not os.path.exists(active_input_db):
        raise FileNotFoundError(f"DB not found: {active_input_db}")

    with connect(active_input_db) as src_db:
        total = src_db.count()
    if total == 0:
        with connect(out_db_path):
            pass
        return out_db_path

    worker_gpu_plan = _build_worker_gpu_plan(
        device=device,
        gpus=gpus,
        workers_per_gpu=workers_per_gpu,
        cpu_workers=cpu_workers,
    )
    if not worker_gpu_plan:
        raise RuntimeError("No workers planned.")

    if cpu_threads_per_worker is None:
        ncpu = os.cpu_count() or 1
        cpu_threads_per_worker = max(1, ncpu // max(1, len(worker_gpu_plan)))
    _set_thread_env(cpu_threads_per_worker)

    if device.lower() != "cpu":
        _prepare_model_assets(model_name, workspace)

    def _spawn_worker(slot_id, attempt, gpu_id, ctx, task_queue, result_queue, stop_event):
        return _spawn_uma_worker(
            slot_id=slot_id,
            attempt=attempt,
            gpu_id=gpu_id,
            ctx=ctx,
            task_queue=task_queue,
            result_queue=result_queue,
            stop_event=stop_event,
            active_input_db=active_input_db,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            workspace=workspace,
            fmax=fmax,
            max_steps=max_steps,
            cpu_threads_per_worker=cpu_threads_per_worker,
            shard_db_dir=shard_db_dir,
            empty_cache_every=empty_cache_every,
        )

    run_state = run_ase_db_task_queue_pool(
        input_db=active_input_db,
        total_tasks=total,
        worker_gpu_plan=worker_gpu_plan,
        spawn_worker=_spawn_worker,
        failure_log_path=fail_log_path,
        progress_desc="Optimizing",
        show_progress=show_progress,
        verbose=verbose,
        queue_size=queue_size,
        warmup_workers_per_gpu=warmup_workers_per_gpu,
        workers_per_gpu=workers_per_gpu,
        ramp_step_per_gpu=ramp_step_per_gpu,
        ramp_interval_sec=ramp_interval_sec,
        stable_window_sec=stable_window_sec,
        gpu_headroom_gb=gpu_headroom_gb,
        init_timeout_sec=init_timeout_sec,
        startup_stagger_sec=startup_stagger_sec,
        global_init_concurrency=global_init_concurrency,
        init_concurrency_per_gpu=init_concurrency_per_gpu,
        max_restarts_per_slot=max_restarts_per_slot,
        restart_backoff_sec=restart_backoff_sec,
        task_timeout_sec=task_timeout_sec,
    )

    merged_rows = 0
    worker_db_paths = list(dict.fromkeys(run_state["spawn_artifacts"]))
    if run_state["normal_shutdown"]:
        _safe_log("[Main] Merging worker shard DBs into final output DB...", show_progress)
        merged_rows = _merge_worker_dbs(
            worker_db_paths=worker_db_paths,
            out_db_path=out_db_path,
            write_xyz=write_xyz,
            out_xyz_dir=out_xyz_dir,
            verbose=verbose,
            show_progress=show_progress,
            dedup_on_source_row_id=True,
        )
        if (not keep_worker_dbs) and os.path.exists(shard_db_dir):
            try:
                shutil.rmtree(shard_db_dir)
            except Exception:
                pass

    ok_count = len(run_state["ok_ids"])
    fail_count = len(run_state["fail_ids"])
    _safe_log(
        f"[Summary] total={total}, ok={ok_count}, fail={fail_count}, merged={merged_rows}, "
        f"feeder_done={run_state['feeder_done']}, output_db={out_db_path}",
        show_progress,
    )
    if fail_count > 0:
        _safe_log(f"[Summary] failed log: {fail_log_path}", show_progress)

    return out_db_path


def main():
    parser = argparse.ArgumentParser(description="FAIRChem Parallel Optimization")

    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Root output directory")
    parser.add_argument("--input-db", default=None, help="Input ASE database")
    parser.add_argument("--smiles", nargs="*", default=None, help="Input SMILES string(s) to optimize")

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name for atom refs cache")

    parser.add_argument("--device", default="cuda", help="Compute device: cuda / cuda:0 / cpu")
    parser.add_argument("--gpus", nargs="*", default=None, help="Physical GPU ids, e.g. --gpus 0 1")
    parser.add_argument("--workers-per-gpu", type=int, default=4, help="Max concurrent worker processes per GPU")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Only used when --device cpu")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=None, help="CPU threads per worker (default: avg split)")
    parser.add_argument("--queue-size", type=int, default=None, help="Task queue size (default: auto)")

    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence criteria")
    parser.add_argument("--steps", type=int, default=200, dest="max_steps", help="Max optimization steps")

    parser.add_argument("--write-xyz", action="store_true", help="Also write optimized xyz files")
    parser.add_argument("--keep-worker-dbs", action="store_true", help="Keep per-worker shard DBs after merge")

    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--no-progress", action="store_false", dest="progress", help="Disable progress bar")
    parser.set_defaults(progress=True)

    parser.add_argument("--warmup-workers-per-gpu", type=int, default=2, help="Initial workers per GPU to start")
    parser.add_argument("--ramp-step-per-gpu", type=int, default=1, help="Workers per GPU added per ramp step")
    parser.add_argument("--ramp-interval-sec", type=float, default=60.0, help="Try ramp every N seconds")
    parser.add_argument("--stable-window-sec", type=float, default=60.0, help="Require no failures in last N seconds before ramp")
    parser.add_argument("--gpu-headroom-gb", type=float, default=13.0, help="Keep GPU headroom when ramping (GB)")

    parser.add_argument("--init-timeout-sec", type=float, default=900.0, help="Kill+respawn if init not ready after N seconds")
    parser.add_argument("--startup-stagger-sec", type=float, default=0.2, help="Non-blocking spawn pacing (seconds)")
    parser.add_argument("--global-init-concurrency", type=int, default=1, help="Max concurrent inits across all GPUs")
    parser.add_argument("--init-concurrency-per-gpu", type=int, default=1, help="Max concurrent inits per GPU")

    parser.add_argument("--max-restarts-per-slot", type=int, default=200, help="Max respawns per slot")
    parser.add_argument("--restart-backoff-sec", type=float, default=2.0, help="Base backoff (exponential) for respawn")
    parser.add_argument("--empty-cache-every", type=int, default=0, help="torch.cuda.empty_cache every N tasks (0 disable)")
    parser.add_argument("--pytorch-alloc-conf", type=str, default=None, help="Set PYTORCH_CUDA_ALLOC_CONF")
    parser.add_argument("--task-timeout-sec", type=float, default=None, help="Kill+respawn if a task exceeds N seconds")

    args = parser.parse_args()

    out_path = entry(
        input_db=args.input_db,
        smiles=args.smiles,
        workspace=args.workspace,
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        cpu_workers=args.cpu_workers,
        cpu_threads_per_worker=args.cpu_threads_per_worker,
        fmax=args.fmax,
        max_steps=args.max_steps,
        verbose=args.verbose,
        show_progress=args.progress,
        queue_size=args.queue_size,
        write_xyz=args.write_xyz,
        keep_worker_dbs=args.keep_worker_dbs,
        warmup_workers_per_gpu=args.warmup_workers_per_gpu,
        ramp_step_per_gpu=args.ramp_step_per_gpu,
        ramp_interval_sec=args.ramp_interval_sec,
        stable_window_sec=args.stable_window_sec,
        gpu_headroom_gb=args.gpu_headroom_gb,
        init_timeout_sec=args.init_timeout_sec,
        startup_stagger_sec=args.startup_stagger_sec,
        global_init_concurrency=args.global_init_concurrency,
        init_concurrency_per_gpu=args.init_concurrency_per_gpu,
        max_restarts_per_slot=args.max_restarts_per_slot,
        restart_backoff_sec=args.restart_backoff_sec,
        empty_cache_every=args.empty_cache_every,
        pytorch_alloc_conf=args.pytorch_alloc_conf,
        task_timeout_sec=args.task_timeout_sec,
    )

    print(f"\nOptimization finished. Output DB: {out_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
