import os
import shutil
import time

import numpy as np
import torch
from ase.db import connect
from dftio.data import _keys
from dptb.data import AtomicData, AtomicDataDict, DataLoader
from dptb.data.build import build_dataset
from dptb.nn.build import build_model
from dptb.nn.hr2hk import HR2HK, HR2HK_Gamma_Only

from emoles.inference.common_tools import (
    extract_model_params,
    get_row_charge,
    get_row_dielectric_constant,
    get_row_identifier_payload,
)
from emoles.utils.db import (
    merge_lmdb_directories,
    open_lmdb_environment,
    put_pickle_record,
    reset_lmdb_directory,
)
from emoles.utils.parallel import write_json_file


default_fine_tune_ckpt_path = r"/share/dptb_ckpt/fine_tune/best.pth"
default_fine_tune_input_json_path = r"/share/dptb_ckpt/fine_tune/input.json"


def _normalize_max_items(max_items, limit):
    if limit is not None:
        return limit
    return max_items


def _build_gamma_projectors(model, device, has_overlap):
    projectors = {
        "hamiltonian": HR2HK_Gamma_Only(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.HAMILTONIAN_KEY,
            overlap=True,
            device=device,
        )
    }
    if has_overlap:
        projectors["overlap"] = HR2HK(
            idp=model.idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            overlap=True,
            device=device,
        )
    return projectors


def _predict_gamma_matrices(batch_info, device, projectors):
    batch_info["kpoint"] = torch.tensor([0.0, 0.0, 0.0], device=device)
    matrices = {}

    ham_out_data = projectors["hamiltonian"].forward(batch_info)
    matrices["hamiltonian"] = (
        ham_out_data[AtomicDataDict.HAMILTONIAN_KEY].real.detach().cpu().numpy()
    )

    if "overlap" in projectors:
        overlap_out_data = projectors["overlap"].forward(batch_info)
        matrices["overlap"] = (
            overlap_out_data[AtomicDataDict.OVERLAP_KEY].real.detach().cpu().numpy()
        )

    return matrices


def _prepare_reference_loader(lmdb_path, basis, r_max):
    reference_info = {
        "root": lmdb_path,
        "prefix": "data",
        "type": "LMDBDataset",
        "get_DM": False,
        "get_Hamiltonian": False,
        "get_overlap": False,
    }
    reference_datasets = build_dataset(
        basis=basis,
        r_max=r_max,
        train_w_charge=True,
        train_w_eps=True,
        **reference_info,
    )
    return DataLoader(dataset=reference_datasets, batch_size=1, shuffle=False)


def _iter_predicted_batches(reference_loader, model, device, max_items):
    for idx, ref_batch in enumerate(reference_loader):
        if max_items is not None and idx >= int(max_items):
            break
        batch = AtomicData.to_AtomicDataDict(ref_batch.to(device))
        with torch.no_grad():
            yield idx, model(batch)


def _build_infer_record(idx, source_metadata, matrices):
    payload = dict(source_metadata or {})
    payload["infer_idx"] = int(idx)
    payload["hamiltonian"] = matrices["hamiltonian"]
    if "overlap" in matrices:
        payload["overlap"] = matrices["overlap"]
    return payload


def _resolve_infer_record_key(idx, source_metadata):
    for key in ("source_row_id", "source_idx"):
        value = (source_metadata or {}).get(key)
        if value is not None:
            return int(value)
    return int(idx)


def _normalize_assignment_item(item, fallback_idx):
    if isinstance(item, dict):
        source_idx = item.get("source_idx", fallback_idx)
        source_row_id = item.get("source_row_id", item.get("row_id"))
        sample_id = item.get("sample_id", source_row_id if source_row_id is not None else source_idx)
        return int(source_idx), int(source_row_id), sample_id

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        source_idx = item[0]
        source_row_id = item[1]
        sample_id = item[2] if len(item) >= 3 else source_row_id
        return int(source_idx), int(source_row_id), sample_id

    raise TypeError(f"Unsupported assignment item: {item!r}")


def ase_db_2_dummy_dptb_lmdb(
    ase_db_path: str,
    dptb_lmdb_path: str,
    txn_batch_size: int = 128,
    items=None,
):
    if os.path.exists(dptb_lmdb_path):
        shutil.rmtree(dptb_lmdb_path)
    os.makedirs(dptb_lmdb_path, exist_ok=True)

    dptb_lmdb_path = os.path.join(dptb_lmdb_path, f"data.{os.getpid()}.lmdb")
    reset_lmdb_directory(dptb_lmdb_path)
    lmdb_env = open_lmdb_environment(dptb_lmdb_path)
    record_index = []
    txn = lmdb_env.begin(write=True)
    try:
        with connect(ase_db_path) as src_db:
            if items is None:
                iterable = (
                    (idx, row, None)
                    for idx, row in enumerate(src_db.select())
                )
            else:
                iterable = (
                    (
                        local_idx,
                        src_db.get(id=_normalize_assignment_item(item, local_idx)[1]),
                        _normalize_assignment_item(item, local_idx),
                    )
                    for local_idx, item in enumerate(items)
                )

            for idx, row, assignment in iterable:
                if assignment is None:
                    source_idx = idx
                    source_row_id = None
                    sample_id = None
                else:
                    source_idx, source_row_id, sample_id = assignment
                    if row is None:
                        raise KeyError(
                            f"ASE row id not found while building DPTB LMDB: {source_row_id}"
                        )

                an_atoms = row.toatoms()
                source_metadata = get_row_identifier_payload(row, fallback_idx=source_idx)
                if assignment is not None:
                    source_metadata.update(
                        {
                            "source_idx": int(source_idx),
                            "source_row_id": int(source_row_id),
                            "sample_id": sample_id,
                        }
                    )
                source_metadata.update(
                    {
                        "charge": get_row_charge(row, 0),
                        "dielectric_constant": get_row_dielectric_constant(row, 0),
                    }
                )
                data_dict = {
                    _keys.ATOMIC_NUMBERS_KEY: an_atoms.numbers,
                    _keys.PBC_KEY: torch.tensor([False, False, False]).numpy(),
                    _keys.POSITIONS_KEY: an_atoms.positions.reshape(1, -1, 3).astype("float32"),
                    _keys.CELL_KEY: an_atoms.cell.reshape(1, 3, 3).astype("float32"),
                    "charge": source_metadata["charge"],
                    "dielectric_constant": source_metadata["dielectric_constant"],
                    "idx": idx,
                    "source_idx": source_metadata["source_idx"],
                    "source_row_id": source_metadata["source_row_id"],
                    "sample_id": source_metadata["sample_id"],
                    "nf": 0,
                }
                put_pickle_record(txn, idx, data_dict)
                record_index.append(source_metadata)
                if (idx + 1) % max(1, int(txn_batch_size)) == 0:
                    txn.commit()
                    txn = lmdb_env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        lmdb_env.close()

    return record_index


def save_info_2_npy(
    folder_path,
    idx,
    batch_info,
    model,
    device,
    has_overlap,
    projectors=None,
):
    if projectors is None:
        projectors = _build_gamma_projectors(model=model, device=device, has_overlap=has_overlap)

    matrices = _predict_gamma_matrices(
        batch_info=batch_info,
        device=device,
        projectors=projectors,
    )
    item_dir = os.path.join(folder_path, str(idx))
    os.makedirs(item_dir, exist_ok=True)
    np.save(os.path.join(item_dir, "predicted.npy"), matrices["hamiltonian"])
    if "overlap" in matrices:
        np.save(os.path.join(item_dir, "predicted_overlap.npy"), matrices["overlap"])


def save_info_2_lmdb(
    txn,
    idx,
    source_metadata,
    batch_info,
    model,
    device,
    has_overlap=False,
    projectors=None,
):
    if projectors is None:
        projectors = _build_gamma_projectors(model=model, device=device, has_overlap=has_overlap)

    matrices = _predict_gamma_matrices(
        batch_info=batch_info,
        device=device,
        projectors=projectors,
    )
    record = _build_infer_record(
        idx=idx,
        source_metadata=source_metadata,
        matrices=matrices,
    )
    record_key = _resolve_infer_record_key(idx=idx, source_metadata=source_metadata)
    put_pickle_record(txn, record_key, record)
    return record


def merge_infer_lmdb_shards(infer_root_path, merged_name="merged.lmdb"):
    infer_root_path = os.path.abspath(infer_root_path)
    shard_paths = []
    for name in sorted(os.listdir(infer_root_path)):
        shard_path = os.path.join(infer_root_path, name)
        if not os.path.isdir(shard_path):
            continue
        if not name.endswith(".lmdb"):
            continue
        if name == merged_name:
            continue
        shard_paths.append(shard_path)

    if not shard_paths:
        raise FileNotFoundError(f"No infer LMDB shards found under: {infer_root_path}")

    merged_path = os.path.join(infer_root_path, merged_name)
    merged_entries = merge_lmdb_directories(
        shard_paths,
        merged_path,
        preserve_keys=True,
    )
    write_json_file(
        os.path.join(merged_path, "manifest.json"),
        {
            "entries": merged_entries,
            "merged_name": merged_name,
            "merged_path": merged_path,
            "source_shards": shard_paths,
            "key_field": "source_row_id",
        },
    )
    return merged_path


def _prepare_dptb_model(checkpoint_path, device):
    import e3nn

    e3nn.set_optimization_defaults(jit_script_fx=False)
    device = torch.device(device)
    model = build_model(checkpoint=checkpoint_path)
    model.to(device)
    model.eval()
    basis, r_max = extract_model_params(model)
    return model, device, basis, r_max


def _prepare_dptb_inference(
    ase_db_path,
    checkpoint_path,
    device,
    input_lmdb_root,
    has_overlap,
):
    model, device, basis, r_max = _prepare_dptb_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    input_records = ase_db_2_dummy_dptb_lmdb(ase_db_path, input_lmdb_root)
    reference_loader = _prepare_reference_loader(
        lmdb_path=input_lmdb_root,
        basis=basis,
        r_max=r_max,
    )
    projectors = _build_gamma_projectors(model=model, device=device, has_overlap=has_overlap)
    return model, device, basis, r_max, input_records, reference_loader, projectors


def dptb_infer_from_ase_db(
    ase_db_path: str,
    out_path: str,
    checkpoint_path: str = default_fine_tune_ckpt_path,
    max_items: int = None,
    device: str = "cuda",
    limit: int = None,
):
    max_items = _normalize_max_items(max_items=max_items, limit=limit)

    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)
    os.makedirs(abs_out_path, exist_ok=True)

    lmdb_path = os.path.join(abs_out_path, "lmdb")
    npy_path = os.path.join(abs_out_path, "results")
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.makedirs(npy_path)

    model, device, _basis, _r_max, _input_records, reference_loader, projectors = _prepare_dptb_inference(
        ase_db_path=ase_db_path,
        checkpoint_path=checkpoint_path,
        device=device,
        input_lmdb_root=lmdb_path,
        has_overlap=False,
    )

    start_time = time.time()
    processed_items = 0
    for idx, predicted_data in _iter_predicted_batches(
        reference_loader=reference_loader,
        model=model,
        device=device,
        max_items=max_items,
    ):
        save_info_2_npy(
            folder_path=npy_path,
            idx=idx,
            batch_info=predicted_data,
            model=model,
            device=device,
            has_overlap=False,
            projectors=projectors,
        )
        processed_items += 1

    end_time = time.time()
    print("DPTB inference done.")
    second_per_item = (end_time - start_time) / max(1, processed_items)
    print(f"DPTB Inference Time (s/item): {second_per_item}")


def dptb_infer_to_lmdb_from_ase_db(
    ase_db_path: str,
    out_path: str,
    checkpoint_path: str = default_fine_tune_ckpt_path,
    max_items: int = None,
    device: str = "cuda",
    limit: int = None,
    infer_dir_name: str = "infer",
    worker_name: str = "worker_0000",
    has_overlap: bool = False,
    merge_shards: bool = False,
    txn_batch_size: int = 32,
    cleanup_input_lmdb: bool = True,
):
    max_items = _normalize_max_items(max_items=max_items, limit=limit)

    abs_out_path = os.path.abspath(out_path)
    ase_db_path = os.path.abspath(ase_db_path)
    infer_root = os.path.join(abs_out_path, infer_dir_name)
    shard_name = str(worker_name or "worker_0000")
    if not shard_name.endswith(".lmdb"):
        shard_name = f"{shard_name}.lmdb"
    worker_root_name = shard_name[:-5]
    input_lmdb_root = os.path.join(abs_out_path, "infer_input_lmdb", worker_root_name)
    shard_path = os.path.join(infer_root, shard_name)

    os.makedirs(abs_out_path, exist_ok=True)
    os.makedirs(infer_root, exist_ok=True)

    model, device, basis, r_max, input_records, reference_loader, projectors = _prepare_dptb_inference(
        ase_db_path=ase_db_path,
        checkpoint_path=checkpoint_path,
        device=device,
        input_lmdb_root=input_lmdb_root,
        has_overlap=has_overlap,
    )

    reset_lmdb_directory(shard_path)
    output_env = open_lmdb_environment(shard_path)
    processed_items = 0
    start_time = time.time()
    txn = output_env.begin(write=True)
    try:
        for idx, predicted_data in _iter_predicted_batches(
            reference_loader=reference_loader,
            model=model,
            device=device,
            max_items=max_items,
        ):
            source_metadata = (
                input_records[idx]
                if idx < len(input_records)
                else {"source_idx": idx, "source_row_id": None, "sample_id": idx}
            )
            save_info_2_lmdb(
                txn=txn,
                idx=idx,
                source_metadata=source_metadata,
                batch_info=predicted_data,
                model=model,
                device=device,
                has_overlap=has_overlap,
                projectors=projectors,
            )
            processed_items += 1
            if processed_items % max(1, int(txn_batch_size)) == 0:
                txn.commit()
                txn = output_env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        output_env.close()

    merged_path = None
    if merge_shards:
        merged_path = merge_infer_lmdb_shards(infer_root)

    if cleanup_input_lmdb and os.path.exists(input_lmdb_root):
        shutil.rmtree(input_lmdb_root)

    end_time = time.time()
    write_json_file(
        os.path.join(shard_path, "manifest.json"),
        {
            "ase_db_path": ase_db_path,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "device": str(device),
            "entries": processed_items,
            "worker_name": worker_name,
            "shard_path": shard_path,
            "infer_root": infer_root,
            "has_overlap": has_overlap,
            "basis": basis,
            "r_max": r_max,
            "second_per_item": (end_time - start_time) / max(1, processed_items),
            "merged_path": merged_path,
            "cleanup_input_lmdb": cleanup_input_lmdb,
            "key_field": "source_row_id",
        },
    )
    write_json_file(
        os.path.join(infer_root, "manifest.json"),
        {
            "infer_root": infer_root,
            "merged_path": merged_path,
            "worker_lmdb_paths": [shard_path],
            "workers": 1,
            "key_field": "source_row_id",
        },
    )
    print("DPTB LMDB inference done.")
    print(f"LMDB shard path: {shard_path}")
    print(f"DPTB Inference Time (s/item): {(end_time - start_time) / max(1, processed_items)}")
    return {
        "infer_root": infer_root,
        "shard_path": shard_path,
        "entries": processed_items,
        "merged_path": merged_path,
    }
