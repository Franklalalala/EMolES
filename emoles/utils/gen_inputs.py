import copy

import torch

from emoles.nn.build import build_model
from emoles.utils.config_e3 import TestFullConfigE3, TrainFullConfigE3


def gen_inputs(mode, task="train", model=None):
    if mode != "e3":
        raise ValueError("EMolES only provides the e3 density-matrix config.")
    if task not in {"train", "test"}:
        raise ValueError("task should be train or test")

    input_dict = copy.deepcopy(
        TrainFullConfigE3 if task == "train" else TestFullConfigE3
    )

    if model is not None:
        if isinstance(model, str):
            model = build_model(checkpoint=model)
        if model.name != "nnenv":
            raise NotImplementedError("EMolES config updates require an NNENV model.")

        dtype = model.dtype if isinstance(model.dtype, str) else str(model.dtype).split(".")[-1]
        device = "cpu" if model.device == "cpu" or model.device == torch.device("cpu") else "cuda"
        input_dict["common_options"].update(
            {
                "basis": model.basis,
                "dtype": dtype,
                "device": device,
                "overlap": hasattr(model, "overlap"),
            }
        )
        input_dict["model_options"].update(model.model_options)

    return input_dict
