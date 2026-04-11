import json
import os
import pickle
import shutil
from typing import Iterable

import lmdb
from ase.db.core import connect


LMDB_MAP_SIZE = 1048576000000
LMDB_KEY_LENGTH = 4


def lmdb_index_key(index):
    return int(index).to_bytes(length=LMDB_KEY_LENGTH, byteorder="big")


def reset_lmdb_directory(lmdb_path):
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path, exist_ok=True)


def open_lmdb_environment(
    lmdb_path,
    readonly=False,
    lock=None,
    map_size=LMDB_MAP_SIZE,
):
    if readonly:
        return lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            subdir=True,
        )

    os.makedirs(lmdb_path, exist_ok=True)
    return lmdb.open(
        lmdb_path,
        map_size=map_size,
        lock=True if lock is None else lock,
        subdir=True,
    )


def put_pickle_record(txn, index, payload):
    return txn.put(
        lmdb_index_key(index),
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
    )


def get_pickle_record(lmdb_path, index, default=None):
    db_env = open_lmdb_environment(lmdb_path, readonly=True)
    try:
        with db_env.begin() as txn:
            payload = txn.get(lmdb_index_key(index))
            if payload is None:
                return default
            return pickle.loads(payload)
    finally:
        db_env.close()


def iter_lmdb_records(lmdb_path):
    db_env = open_lmdb_environment(lmdb_path, readonly=True)
    try:
        with db_env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                yield int.from_bytes(key, byteorder="big"), pickle.loads(value)
    finally:
        db_env.close()


def resolve_lmdb_paths(lmdb_root_or_path, prefer_merged=True):
    abs_path = os.path.abspath(lmdb_root_or_path)
    if abs_path.endswith(".lmdb"):
        return [abs_path]
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"LMDB path not found: {abs_path}")

    manifest_path = os.path.join(abs_path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f_obj:
            manifest = json.load(f_obj)
        manifest_merged_path = manifest.get("merged_path")
        if manifest_merged_path:
            merged_path = os.path.abspath(manifest_merged_path)
            if prefer_merged and os.path.isdir(merged_path):
                return [merged_path]

        worker_lmdb_paths = [
            os.path.abspath(path)
            for path in manifest.get("worker_lmdb_paths", [])
            if path and os.path.isdir(os.path.abspath(path))
        ]
        if worker_lmdb_paths:
            return worker_lmdb_paths

    lmdb_paths = []
    for name in sorted(os.listdir(abs_path)):
        lmdb_path = os.path.join(abs_path, name)
        if os.path.isdir(lmdb_path) and name.endswith(".lmdb"):
            lmdb_paths.append(lmdb_path)
    if not lmdb_paths:
        raise FileNotFoundError(f"No LMDB directories found under: {abs_path}")
    return lmdb_paths


def get_pickle_record_any(lmdb_root_or_path, index, default=None):
    for lmdb_path in resolve_lmdb_paths(lmdb_root_or_path):
        payload = get_pickle_record(lmdb_path, index, default=None)
        if payload is not None:
            return payload
    return default


def extract_lmdb_records(lmdb_root_or_path, target_lmdb_path, record_keys):
    target_keys = {int(key) for key in record_keys if key is not None}
    reset_lmdb_directory(target_lmdb_path)
    if not target_keys:
        return {"copied": 0, "missing": []}

    copied = 0
    output_env = open_lmdb_environment(target_lmdb_path)
    txn = output_env.begin(write=True)
    try:
        for lmdb_path in resolve_lmdb_paths(lmdb_root_or_path):
            if not target_keys:
                break
            for key, payload in iter_lmdb_records(lmdb_path):
                if key not in target_keys:
                    continue
                put_pickle_record(txn, key, payload)
                copied += 1
                target_keys.remove(key)
                if copied % 128 == 0:
                    txn.commit()
                    txn = output_env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        output_env.close()

    return {"copied": copied, "missing": sorted(target_keys)}


def merge_lmdb_directories(source_lmdb_paths: Iterable, merged_lmdb_path, preserve_keys=False):
    reset_lmdb_directory(merged_lmdb_path)
    merged_env = open_lmdb_environment(merged_lmdb_path)
    merged_entries = 0
    try:
        with merged_env.begin(write=True) as merged_txn:
            for source_lmdb_path in source_lmdb_paths:
                if not os.path.exists(source_lmdb_path):
                    continue
                for key, payload in iter_lmdb_records(source_lmdb_path):
                    target_key = key if preserve_keys else merged_entries
                    success = put_pickle_record(
                        merged_txn,
                        target_key,
                        payload,
                    )
                    if preserve_keys and not success:
                        raise KeyError(f"Duplicate LMDB record key during merge: {key}")
                    merged_entries += 1
    finally:
        merged_env.close()
    return merged_entries


def get_row_data(row):
    try:
        data = getattr(row, "data", None)
        return dict(data) if data else {}
    except Exception:
        return {}


def get_row_key_value_pairs(row):
    try:
        key_value_pairs = getattr(row, "key_value_pairs", None)
        return dict(key_value_pairs) if key_value_pairs else {}
    except Exception:
        return {}


def merge_row_metadata(row):
    meta = {}
    meta.update(get_row_key_value_pairs(row))
    meta.update(get_row_data(row))
    dielectric_constant = getattr(row, "dielectric_constant", None)
    if dielectric_constant is not None and "dielectric_constant" not in meta:
        meta["dielectric_constant"] = dielectric_constant
    return meta


def prepare_ase_db_worker_shards(
    source_db_path,
    work_root,
    n_workers,
    global_n_save_cube_items=0,
    max_items=None,
):
    if os.path.exists(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root, exist_ok=True)

    shard_specs = []
    shard_dbs = []
    for worker_id in range(int(n_workers)):
        worker_name = f"worker_{worker_id:02d}"
        worker_root = os.path.join(work_root, worker_name)
        os.makedirs(worker_root, exist_ok=True)
        shard_db_path = os.path.join(worker_root, "input.db")
        shard_specs.append(
            {
                "worker_id": int(worker_id),
                "worker_name": worker_name,
                "worker_root": worker_root,
                "shard_db_path": shard_db_path,
                "local_to_global": [],
                "num_items": 0,
                "n_save_cube_items": 0,
            }
        )
        shard_dbs.append(connect(shard_db_path))

    try:
        with connect(source_db_path) as src_db:
            total_rows = src_db.count()
            for source_idx, row in enumerate(src_db.select()):
                if max_items is not None and source_idx >= int(max_items):
                    break

                worker_id = source_idx % int(n_workers)
                shard_spec = shard_specs[worker_id]
                shard_db = shard_dbs[worker_id]

                atoms = row.toatoms()
                data = merge_row_metadata(row)
                data["source_idx"] = int(source_idx)
                data["source_row_id"] = int(row.id)

                shard_db.write(atoms, data=data)
                shard_spec["local_to_global"].append(int(source_idx))
                shard_spec["num_items"] += 1
                if source_idx < int(global_n_save_cube_items):
                    shard_spec["n_save_cube_items"] += 1

        final_specs = []
        for shard_spec in shard_specs:
            if shard_spec["num_items"] <= 0:
                continue
            index_map_path = os.path.join(shard_spec["worker_root"], "index_map.json")
            shard_spec["index_map_path"] = index_map_path
            with open(index_map_path, "w", encoding="utf-8") as f_obj:
                json.dump(
                    {
                        "worker_id": shard_spec["worker_id"],
                        "worker_name": shard_spec["worker_name"],
                        "local_to_global": shard_spec["local_to_global"],
                        "num_items": shard_spec["num_items"],
                        "n_save_cube_items": shard_spec["n_save_cube_items"],
                        "total_rows": total_rows,
                        "max_items": max_items,
                    },
                    f_obj,
                    indent=2,
                )
            final_specs.append(shard_spec)
        return final_specs
    finally:
        for shard_db in shard_dbs:
            try:
                shard_db.close()
            except Exception:
                pass


def update_ase_db_w_lmdb(
    src_ase_db_path,
    dump_ase_db_path,
    lmdb_path,
    update_keys_list=None,
):
    if update_keys_list is None:
        update_keys_list = ["iterations", "total_time"]

    db_env = open_lmdb_environment(lmdb_path, readonly=True)
    with db_env.begin() as txn, connect(dump_ase_db_path) as dump_db, connect(
        src_ase_db_path
    ) as src_db:
        stat = txn.stat()
        entries = stat["entries"]
        print(f"lmdb counts: {entries}")
        db_counts = src_db.count()
        print(f"src ase db counts: {db_counts}")
        min_counts = min(db_counts, entries)
        for idx in range(min_counts):
            data_dict = txn.get(lmdb_index_key(idx))
            data_dict = pickle.loads(data_dict)
            row = src_db.get(id=idx + 1)
            old_data = row.data
            old_atoms = row.toatoms()
            for key in update_keys_list:
                old_data.update({key: data_dict[key]})
            dump_db.write(old_atoms, data=old_data)
    db_env.close()
