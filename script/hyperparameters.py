import numpy as np

# string_type = ["smiles", "selfies", "deepsmiles", "modify_deepsmiles", "modify_smiles"][4]
encoder_type = ["CNN", "RNN"][1]

def load_params(type = "None"):
    if type == "None":
        pass
    else:
        string_type = type
    parameters = {
        "data_file": "../activity_coefficient_v1_gamma.csv",

        "do_prop_pred": True,
        "reg_prop_tasks": ["log_gamma_1", "log_gamma_2"],

        "rand_seed": 21,
        "max_len": 20,
        "batch_size": 64,
        "val_split": 0.1,

        "hidden_dim": 128,
        "dropout_rate": 0.08,

        "epochs": 500,
        "lr": 0.001,
        "momentum": 0.99,
        "loss": "categorical_crossentropy",
        "reg_prop_pred_loss": "mse",

        "anneal_sigmod_slope": 0.5,
        "vae_annealer_start": 100

    }


    new_params = {
        "string_type": string_type,
        "dict_path": "../models/" + string_type + "/dict_" + string_type + ".json",
        "history_file": "../models/" + string_type + "/history.csv",
        "encoder_weights_file": "../models/" + string_type + "/encoder.h5",
        "decoder_weights_file": "../models/" + string_type + "/decoder.h5",
        "prop_pred_weights_file": "../models/" + string_type + "/prop_pred.h5",
        "checkpoint_path": "../models/" + string_type + "/",
        "test_idx_file": "../models/" + string_type + "/test_idx.npy",
        "train_idx_file": "../models/" + string_type + "/train_idx.npy",
        "discard_idx_file": "../models/" + string_type + "/t_discard_idx.npy",
        "used_data_path": "../models/" + string_type + "/used_data.csv",

        "encoder_type": encoder_type,
    }
    parameters.update(new_params)
    return parameters

