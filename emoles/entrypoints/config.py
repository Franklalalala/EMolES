import json
import logging
import os
from typing import Optional

from emoles.utils.gen_inputs import gen_inputs

__all__ = ["get_full_config", "config"]


def get_full_config(model=None, train=True, e3tb=True):
    if not train:
        raise ValueError("EMolES currently exposes training config generation only.")
    if not e3tb:
        raise ValueError("EMolES config generation requires --e3tb.")
    return "train_E3", gen_inputs(mode="e3", task="train", model=model)


def config(
    PATH: str,
    train: bool = True,
    test: bool = False,
    e3tb: bool = False,
    model: str = None,
    log_level: int = logging.INFO,
    log_path: Optional[str] = None,
    **kwargs,
):
    if test:
        raise ValueError("EMolES test config generation is not part of this package.")
    name, full_config = get_full_config(model=model, train=train, e3tb=e3tb)

    if not PATH.endswith(".json"):
        PATH = os.path.join(PATH, "input_template.json")

    with open(PATH, "w", encoding="utf-8") as fp:
        logging.info("Writing %s config to %s", name, PATH)
        json.dump(full_config, fp, indent=4)

    return 0
