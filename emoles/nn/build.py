import copy
import logging

import torch

from emoles.nn.model import NNENV
from emoles.utils.tools import j_loader

log = logging.getLogger(__name__)


def build_model(
    checkpoint: str = None,
    model_options: dict = None,
    common_options: dict = None,
    no_check: bool = False,
    device: str = None,
):
    """Build an EMolES density-matrix model.

    The standalone EMolES library keeps the E3/EMolES density-matrix training
    path as its public build mode.
    """

    model_options = {} if model_options is None else copy.deepcopy(model_options)
    common_options = {} if common_options is None else copy.deepcopy(common_options)

    if checkpoint is not None:
        if checkpoint.split(".")[-1] == "json":
            ckpt_config = j_loader(checkpoint)
        else:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ckpt_config = ckpt["config"]
            del ckpt

        if not model_options:
            model_options = copy.deepcopy(ckpt_config["model_options"])
        if not common_options:
            common_options = copy.deepcopy(ckpt_config["common_options"])
    elif not (model_options and common_options):
        raise ValueError(
            "model_options and common_options are required when building EMolES "
            "from scratch."
        )

    if not (model_options.get("embedding") and model_options.get("prediction")):
        raise ValueError(
            "EMolES model_options must define both `embedding` and `prediction`."
        )
    if model_options["prediction"].get("method") != "e3tb":
        raise ValueError("EMolES supports the e3tb prediction method.")
    if model_options["embedding"].get("method") not in {"emoles", "emoles_openequi"}:
        log.warning(
            "Building a non-EMolES embedding method from compatibility config: %s",
            model_options["embedding"].get("method"),
        )

    if device:
        common_options["device"] = device

    if checkpoint is None:
        model = NNENV(**model_options, **common_options)
    else:
        model = NNENV.from_reference(checkpoint, **model_options, **common_options)

    if not no_check:
        for key, value in model.model_options.items():
            if key not in model_options:
                log.warning("model_options.%s missing in input; using %r", key, value)
            else:
                deep_dict_difference(key, value, model_options)

    model.to(model.device)
    return model


def deep_dict_difference(base_key, expected_value, model_options):
    target_dict = copy.deepcopy(model_options)
    if isinstance(expected_value, dict):
        for sub_key, sub_value in expected_value.items():
            if sub_key not in target_dict.get(base_key, {}):
                log.warning(
                    "model option %s.%s missing in input; using %r",
                    base_key,
                    sub_key,
                    sub_value,
                )
            else:
                deep_dict_difference(sub_key, sub_value, target_dict[base_key])
    elif expected_value != target_dict[base_key]:
        log.warning(
            "model option %s is %r in the built model, but %r in input",
            base_key,
            expected_value,
            target_dict[base_key],
        )
