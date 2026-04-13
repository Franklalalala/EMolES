import os
import queue as pyqueue
import sys
import time
from pathlib import Path

from ase import Atoms
from ase.db import connect


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _fake_queue_worker(slot_id, attempt, gpu_id, artifact_path, task_queue, result_queue, stop_event):
    pid = os.getpid()
    device_name = "cpu" if gpu_id is None else f"cuda:{gpu_id}"
    result_queue.put(
        {
            "type": "worker_ready",
            "slot_id": int(slot_id),
            "attempt": int(attempt),
            "pid": int(pid),
            "device": device_name,
            "init_s": 0.0,
            "mem_alloc_mb": None,
            "mem_reserved_mb": None,
        }
    )

    with open(artifact_path, "a", encoding="utf-8") as f_obj:
        while True:
            try:
                row_id = task_queue.get(timeout=0.2)
            except pyqueue.Empty:
                if stop_event.is_set():
                    break
                continue

            if row_id is None:
                break

            row_id = int(row_id)
            result_queue.put(
                {
                    "type": "started",
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                    "row_id": row_id,
                    "name": f"id_{row_id:06d}",
                }
            )
            f_obj.write(f"{row_id}\n")
            f_obj.flush()
            time.sleep(0.01)
            result_queue.put(
                {
                    "type": "done",
                    "slot_id": int(slot_id),
                    "attempt": int(attempt),
                    "pid": int(pid),
                    "device": device_name,
                    "row_id": row_id,
                    "name": f"id_{row_id:06d}",
                }
            )

    result_queue.put(
        {
            "type": "worker_exit",
            "slot_id": int(slot_id),
            "attempt": int(attempt),
            "pid": int(pid),
            "device": device_name,
        }
    )


def test_queue_runner_processes_small_ase_db_once(tmp_path):
    from emoles.utils.parallel import run_ase_db_task_queue_pool

    db_path = tmp_path / "tiny.db"
    with connect(db_path) as db:
        for idx in range(4):
            db.write(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74 + idx * 0.01]]), tag=idx)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    def _spawn_worker(slot_id, attempt, gpu_id, ctx, task_queue, result_queue, stop_event):
        artifact_path = artifact_root / f"slot_{slot_id:02d}__a{attempt:02d}.txt"
        proc = ctx.Process(
            target=_fake_queue_worker,
            args=(
                slot_id,
                attempt,
                gpu_id,
                str(artifact_path),
                task_queue,
                result_queue,
                stop_event,
            ),
        )
        proc.start()
        return proc, str(artifact_path)

    run_state = run_ase_db_task_queue_pool(
        input_db=str(db_path),
        total_tasks=4,
        worker_gpu_plan=[None, None],
        spawn_worker=_spawn_worker,
        failure_log_path=str(tmp_path / "failed.log"),
        progress_desc="tiny",
        show_progress=False,
        verbose=False,
        workers_per_gpu=2,
        warmup_workers_per_gpu=2,
        max_restarts_per_slot=8,
    )

    assert run_state["normal_shutdown"] is True
    assert run_state["feeder_done"] is True
    assert run_state["ok_ids"] == [1, 2, 3, 4]
    assert run_state["fail_ids"] == []

    processed = []
    for artifact_path in run_state["spawn_artifacts"]:
        with open(artifact_path, "r", encoding="utf-8") as f_obj:
            processed.extend(int(line.strip()) for line in f_obj if line.strip())

    assert sorted(processed) == [1, 2, 3, 4]
