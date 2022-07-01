import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json, time, random, csv
import numpy as np
import pandas as pd
from functools import partial

import tensorflow as tf
from keras.layers import Lambda
from keras.models import Model, load_model
from keras.metrics import categorical_accuracy, mae
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras import backend as K

import hyperparameters
import mol_utils as mu
import mol_callbacks as mol_cb

from models import encoder_model
from models import decoder_model
from models import property_predictor_model
from models import variational_layers
from tgru_k2_gpu import TerminalGRU

import warnings
warnings.filterwarnings("ignore")


def vectorize_data(params):
    MAX_LEN = params["max_len"]
    CHAR_INDICES = json.load(open(params["dict_path"]))
    NCHARS = len(CHAR_INDICES)
    params['nchars'] = NCHARS
    INDICES_CHAR = dict((CHAR_INDICES[i], i) for i in CHAR_INDICES)
    print(INDICES_CHAR)

    index, names, smiles, strings, targets = mu.load_string_and_data_df(params["data_file"],
                                                                        MAX_LEN,
                                                                        params["string_type"],
                                                                        params["used_data_path"],
                                                                        reg_tasks=params["reg_prop_tasks"])

    print("Training set size is", len(strings))
    print("first string: \"", strings[0], "\"")
    print("total chars:", NCHARS)

    print("Vectorization...")
    X = mu.string_to_hot(strings, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])

    total_size = np.shape(X)[0] // params["batch_size"] * params["batch_size"]
    print("Data size fitting batch size:", total_size)

    TRAIN_FRAC = 1 - 2 * params["val_split"]
    train_size = int(total_size * TRAIN_FRAC)

    if train_size % params["batch_size"] != 0:
        train_size = train_size // params["batch_size"] * params["batch_size"]
        valid_size = int((total_size - train_size) / 2)
    print("Training size:", train_size, "/ Validation size:", valid_size)

    rand_idx = np.arange(X.shape[0])
    np.random.shuffle(rand_idx)

    train_idx, valid_idx, test_idx = rand_idx[:int(train_size)], \
                                     rand_idx[int(train_size):int(train_size + valid_size)], \
                                     rand_idx[int(train_size + valid_size):int(total_size)]
    disc_idx = list(set(np.arange(X.shape[0])).difference(set(list(train_idx) + list(valid_idx) + list(test_idx))))
    index_train, index_valid, index_test, index_disc = [index[i] for i in train_idx], \
                                                       [index[i] for i in valid_idx], \
                                                       [index[i] for i in test_idx], \
                                                       [index[i] for i in disc_idx]
    names_train, names_valid, names_test, names_disc = [names[i] for i in train_idx], \
                                                       [names[i] for i in valid_idx], \
                                                       [names[i] for i in test_idx], \
                                                       [names[i] for i in disc_idx]
    smiles_train, smiles_valid, smiles_test, smiles_disc = [smiles[i] for i in train_idx], \
                                                           [smiles[i] for i in valid_idx], \
                                                           [smiles[i] for i in test_idx], \
                                                           [smiles[i] for i in disc_idx]
    X_train, X_valid, X_test, X_disc = X[train_idx], X[valid_idx], X[test_idx], X[disc_idx]
    Y_train, Y_valid, Y_test, Y_disc = targets[train_idx], targets[valid_idx], targets[test_idx], targets[disc_idx]
    print("first string after shuffle: \"", strings[rand_idx[0]], "\"")

    data_path = params["checkpoint_path"] + "Data_modeling.csv"
    with open(data_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "smiles", "y1", "y2", "set"])
    with open(data_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i, name, smile, y in zip(index_train, names_train, smiles_train, Y_train):
            writer.writerow([i, name, smile, y[0], y[1], "train"])
    with open(data_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i, name, smile, y in zip(index_valid, names_valid, smiles_valid, Y_valid):
            writer.writerow([i, name, smile, y[0], y[1], "valid"])
    with open(data_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i, name, smile, y in zip(index_test, names_test, smiles_test, Y_test):
            writer.writerow([i, name, smile, y[0], y[1], "test"])
    with open(data_path, "a", newline="") as f:
        writer = csv.writer(f)
        for i, name, smile, y in zip(index_disc, names_disc, smiles_disc, Y_disc):
            writer.writerow([i, name, smile, y[0], y[1], "discard"])
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, (index_train, index_valid, index_test)


def load_models(params):
    def identity(x):
        return K.identity(x)

    kl_loss_var = K.variable(1.)

    encoder = encoder_model(params)
    decoder = decoder_model(params)

    x_in = encoder.inputs[0]
    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    x_out = decoder([z_samp, x_in])
    x_out = Lambda(identity, name="x_pred")(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    property_predictor = property_predictor_model(params)
    reg_prop_pred = property_predictor(z_mean)
    reg_prop_pred = Lambda(identity, name="reg_prop_pred")(reg_prop_pred)
    model_outputs.append(reg_prop_pred)

    AE_PP_model = Model(x_in, model_outputs)
    return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print("x_mean shape in kl_loss: ", x_mean.get_shape())
    kl_loss = - 0.5 * \
              K.mean(1 + x_log_var - K.square(x_mean) -
                     K.exp(x_log_var), axis=-1)
    return kl_loss


def main_property_run(params):
    seed_value = params["rand_seed"]
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    start_time = time.time()
    K.clear_session()

    # load data
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, index = vectorize_data(params)
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)

    print(encoder.summary())
    print(decoder.summary())
    print(AE_PP_model.summary())
    print("Random seed is", params["rand_seed"])

    # K.set_learning_phase(0)
    # print(K.learning_phase())

    model_train_targets = {"x_pred": X_train,
                           "z_mean_log_var": np.zeros((np.shape(X_train)[0], params["hidden_dim"] * 2))}
    model_valid_targets = {"x_pred": X_valid,
                           "z_mean_log_var": np.ones((np.shape(X_valid)[0], params["hidden_dim"] * 2))}
    model_losses = {"x_pred": params["loss"], "z_mean_log_var": kl_loss}
    model_loss_weights = {"x_pred": 1., "z_mean_log_var": kl_loss_var}

    if ("reg_prop_tasks" in params) and (len(params["reg_prop_tasks"]) > 0):
        model_train_targets["reg_prop_pred"] = Y_train
        model_valid_targets["reg_prop_pred"] = Y_valid
        model_losses["reg_prop_pred"] = params["reg_prop_pred_loss"]
        model_loss_weights["reg_prop_pred"] = 0.5

    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params["anneal_sigmod_slope"],
                               start=params["vae_annealer_start"])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(vae_sig_schedule, kl_loss_var, 1, "vae")
    csv_callback = CSVLogger(params["history_file"], append=False)
    callbacks = [vae_anneal_callback, csv_callback]

    # callbacks.append(mol_cb.LossWeightTrack(model_loss_weights))
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = 0

    if "checkpoint_path" in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                                                         params=params, prop_pred_model=property_predictor,
                                                         save_best_only=True))

    optim = Adam(lr=params["lr"], beta_1=params["momentum"])
    AE_PP_model.compile(loss=model_losses,
                        loss_weights=model_loss_weights,
                        optimizer=optim,
                        metrics={'x_pred': ['categorical_accuracy', vae_anneal_metric], 'reg_prop_pred': ['mae']})

    AE_PP_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=0,
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[X_valid, model_valid_targets],
                    shuffle=False
                    )

    # encoder.save(params["encoder_weights_file"])
    # decoder.save(params["decoder_weights_file"])
    # property_predictor.save(params["prop_pred_weights_file"])
    encoder = load_model(os.path.join(params["checkpoint_path"], "encoder_{}.h5".format("best")))
    decoder = load_model(os.path.join(params["checkpoint_path"], "decoder_{}.h5".format("best")),
                         custom_objects={'TerminalGRU': TerminalGRU})
    predictor = load_model(os.path.join(params["checkpoint_path"], "prop_pred_{}.h5".format("best")))

    pred_path = params["checkpoint_path"] + "Data_prediction.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y1", "y2", "f1", "f2", "set"])

    def evaluation(X, Y, index=None, set=None):
        z = encoder(tf.constant(X, dtype=tf.float32))
        z = [z[0], K.variable(X)]
        X_r = K.eval(decoder(z))
        accuracy = categorical_accuracy(K.variable(X, dtype=tf.float32),
                                        K.variable(X_r, dtype=tf.float32))
        pred_props = predictor(z[0])
        MAE = mae(Y, pred_props)
        pred_props = K.eval(pred_props)
        with open(pred_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(len(index)):
                writer.writerow([index[i], Y[i][0], Y[i][1], pred_props[i][0], pred_props[i][1], set])
        return np.mean(K.eval(accuracy)), np.mean(K.eval(MAE))

    K.set_learning_phase(0)
    print("Learning phase:", K.learning_phase())
    print("Train accuracy and MAE:", evaluation(X_train, Y_train, index[0], "train"))
    print("Test accuracy and MAE:", evaluation(X_test, Y_test, index[1], "test"))
    print("Valid accuracy and MAE:", evaluation(X_valid, Y_valid, index[2], "valid"))

    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")

    return


if __name__ == "__main__":
    params = hyperparameters.load_params("modify_smiles")
    print("All params:", params)
    main_property_run(params)
