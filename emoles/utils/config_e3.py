TrainFullConfigE3 = {
    "common_options": {
        "basis": {
            "Li": "1s",
            "C": "1s1p",
            "H": "1s",
            "O": "1s1p",
            "F": "1s1p",
            "S": "1s1p1d",
        },
        "device": "cuda",
        "dtype": "float32",
        "overlap": False,
    },
    "model_options": {
        "embedding": {
            "method": "emoles",
            "r_max": 5.0,
            "irreps_hidden": "32x0e+32x1o+16x2e",
            "n_layers": 3,
            "n_radial_basis": 10,
            "avg_num_neighbors": 32,
            "tp_radial_emb": True,
            "use_layer_onehot_tp": True,
            "use_out_onehot_tp": True,
            "self_mix_flag": True,
            "self_mix_type": "edge",
        },
        "prediction": {
            "method": "e3tb",
            "neurons": [64, 64],
        },
    },
    "train_options": {
        "num_epoch": 1500,
        "batch_size": 1,
        "optimizer": {
            "type": "Adam",
            "lr": 0.005,
        },
        "lr_scheduler": {
            "type": "rop",
            "factor": 0.8,
            "patience": 50,
            "min_lr": 1e-6,
        },
        "loss_options": {
            "train": {
                "method": "hamil_abs",
            },
        },
        "save_freq": 100,
        "validation_freq": 10,
        "display_freq": 1,
    },
    "data_options": {
        "train": {
            "root": "./data",
            "prefix": "train",
            "type": "LMDBDataset",
            "get_Hamiltonian": True,
            "get_DM": True,
            "get_overlap": False,
        },
    },
}

TestFullConfigE3 = {}
