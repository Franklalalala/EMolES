"""Small EMolES configuration helpers.

The standalone package keeps the density-matrix training path only.  This file
therefore normalizes the common/train/data/model sections needed by that path
instead of carrying the full historical configuration schema.
"""

from __future__ import annotations

import copy
import logging
from numbers import Number
from typing import Any, Dict, MutableMapping, Tuple, Union


log = logging.getLogger(__name__)


DEFAULT_COMMON_OPTIONS = {
    "seed": 123,
    "device": "cuda",
    "dtype": "float32",
    "overlap": False,
    "orthogonal": False,
    "has_soc": False,
}

DEFAULT_TRAIN_OPTIONS = {
    "num_epoch": 1,
    "batch_size": 1,
    "val_batch_size": 1,
    "ref_batch_size": 1,
    "update_lr_per_iter": False,
    "use_tensorboard": False,
    "valid_fast": True,
    "sliding_win_size": 100,
    "monitor_flag": False,
    "save_freq": 1000,
    "validation_freq": 1000,
    "display_freq": 100,
    "max_ckpt": 4,
    "optimizer": {
        "type": "Adam",
        "lr": 1e-3,
    },
    "lr_scheduler": {
        "type": "rop",
        "factor": 0.8,
        "patience": 50,
        "min_lr": 1e-6,
    },
    "loss_options": {
        "train": {"method": "hamil_abs"},
        "validation": {"method": "hamil_abs"},
        "reference": {"method": "hamil_abs"},
    },
}


def _deep_merge_defaults(target: MutableMapping[str, Any], defaults: Dict[str, Any]) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                target[key] = copy.deepcopy(value)
            else:
                _deep_merge_defaults(existing, value)
        else:
            target.setdefault(key, copy.deepcopy(value))


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an EMolES training config in place-compatible form."""

    normalized = copy.deepcopy(data)
    normalized.setdefault("common_options", {})
    normalized.setdefault("train_options", {})
    normalized.setdefault("data_options", {})
    normalized.setdefault("model_options", {})

    _deep_merge_defaults(normalized["common_options"], DEFAULT_COMMON_OPTIONS)
    _deep_merge_defaults(normalized["train_options"], DEFAULT_TRAIN_OPTIONS)

    model_options = normalized["model_options"]
    if model_options:
        embedding = model_options.get("embedding")
        prediction = model_options.get("prediction")
        if not isinstance(embedding, dict) or not isinstance(prediction, dict):
            raise ValueError("EMolES model_options must contain embedding and prediction dictionaries.")
        embedding.setdefault("method", "emoles")
        prediction.setdefault("method", "e3tb")
        if prediction["method"] != "e3tb":
            raise ValueError("Standalone EMolES supports prediction.method='e3tb'.")

    loss_options = normalized["train_options"]["loss_options"]
    if "validation" not in loss_options and "train" in loss_options:
        loss_options["validation"] = copy.deepcopy(loss_options["train"])
    if "reference" not in loss_options and "train" in loss_options:
        loss_options["reference"] = copy.deepcopy(loss_options["train"])

    return normalized


def normalize_setinfo(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(data)
    for key in ("nframes", "pos_type", "pbc"):
        if key not in normalized:
            raise ValueError(f"Dataset info.json is missing required key: {key}")
    normalized.setdefault("natoms", -1)
    return normalized


def normalize_lmdbsetinfo(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(data)
    normalized.setdefault("r_max", 4.0)
    return normalized


def get_cutoffs_from_model_options(
    model_options: Dict[str, Any],
) -> Tuple[Union[float, int, dict], Any, Any]:
    embedding = model_options.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError("EMolES model_options.embedding is required to collect cutoffs.")

    method = embedding.get("method", "emoles")
    if method not in {"emoles", "emoles_openequi"}:
        log.warning("Collecting cutoffs for compatibility embedding method %s", method)

    r_max = embedding.get("r_max")
    if r_max is None:
        raise ValueError("EMolES model_options.embedding.r_max is required.")
    return r_max, embedding.get("er_max"), embedding.get("oer_max")


def collect_cutoffs(jdata: Dict[str, Any]) -> Dict[str, Any]:
    r_max, er_max, oer_max = get_cutoffs_from_model_options(jdata["model_options"])
    cutoff_options = {"r_max": r_max, "er_max": er_max, "oer_max": oer_max}
    log.info("EMolES cutoff options: %s", cutoff_options)
    return cutoff_options


def chk_avg_per_iter(jdata: Dict[str, Any]) -> bool:
    scheduler = jdata["train_options"]["lr_scheduler"]
    return scheduler.get("type") == "rop" and bool(jdata["train_options"]["update_lr_per_iter"])


def format_cuts(
    rcut: Union[Dict[str, Number], Number],
    decay_w: Number,
    nbuffer: int,
) -> Union[Dict[str, Number], Number]:
    if not isinstance(decay_w, Number) or decay_w <= 0:
        raise ValueError("decay_w should be a positive number")
    buffer_addition = decay_w * nbuffer
    if isinstance(rcut, dict):
        return {key: value + buffer_addition for key, value in rcut.items()}
    if isinstance(rcut, Number):
        return rcut + buffer_addition
    raise TypeError("rcut should be a dict or a number")
