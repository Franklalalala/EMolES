from typing import Optional

from emoles.utils.tools import j_loader, j_must_have
from emoles.utils.argcheck import normalize


def check_config_train(INPUT, init_model: Optional[str], restart: Optional[str], **kwargs):
    if init_model and restart:
        raise RuntimeError("--init-model and --restart cannot be set together")

    jdata = normalize(j_loader(INPUT))

    if not (restart or init_model):
        j_must_have(jdata, "model_options")
        j_must_have(jdata, "train_options")
        model_options = jdata["model_options"]
        if not (model_options.get("embedding") and model_options.get("prediction")):
            raise ValueError(
                "EMolES training requires model_options.embedding and "
                "model_options.prediction."
            )
        if model_options["prediction"].get("method") != "e3tb":
            raise ValueError("EMolES training requires prediction.method='e3tb'.")

    j_must_have(jdata, "data_options")
    assert j_must_have(jdata["data_options"], "train"), (
        "data_options.train is required in the input configuration."
    )

    train_data_config = jdata["data_options"]["train"]
    if train_data_config.get("get_eigenvalues") and not train_data_config.get("get_Hamiltonian"):
        assert jdata["train_options"]["loss_options"]["train"].get("method") in ["eigvals"]
