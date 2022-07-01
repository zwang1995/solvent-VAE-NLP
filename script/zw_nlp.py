import pyomo
from pyomo.environ import *
import numpy as np
import pandas as pd
import time
import json
import csv
from rdkit import Chem

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras import backend as K
K.set_learning_phase(0)
from keras.models import load_model

import hyperparameters
from tgru_k2_gpu import TerminalGRU
import mol_utils as mu
from zw_smiles_split import modify_smiles


def mean_var(path):
    Zs = np.load(path)
    return Zs

def reverse(z):
    return 1/z

def tanh(z):
    return (1 - 2 / (exp(2 * z) + 1)) #(exp(2*z)-1)/(exp(2*z)+1)

def sigmoid(z):
    return (1/(1+exp(-z)))

def softplus(z):
    return (log(1+exp(z)))

def z_to_y(z):
    z = (z - min) / scale
    z = [z[i] * scale[i] + min[i] for i in range(len(z))]

    z = np.array([sum(z[i] * w[i] for i in range(len(z))) for w in w1])
    z = [a + b for a, b in zip(z, b1)]
    z = [sigmoid(i) for i in z]

    z = np.array([sum(z[i] * w[i] for i in range(len(z))) for w in w2])
    z = [a + b for a, b in zip(z, b2)]
    z = [softplus(i) for i in z]

    # z = (z - mu2) / (np.sqrt(std2 + 0.001)) * gamma2 + beta2

    z = np.array([sum(z[i] * w[i] for i in range(len(z))) for w in w3])
    ln_y = [a + b for a, b in zip(z, b3)]
    print(ln_y)

    y = np.exp(ln_y)
    print(y)

    return ("ln_gamma:", list(ln_y), "gamma:", list(y), "Selectivity:", y[1]/y[0], "Capacity:", 1/y[0])

def smiles_to_y(smiles):
    smiles = [smiles]
    smiles = [modify_smiles(i) for i in smiles]
    print(smiles)
    X = mu.string_to_hot_sampling(smiles, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
    X = X.reshape(-1, MAX_LEN, NCHARS)
    z = encoder(tf.constant(X, dtype=tf.float32))
    pred_props = predictor(z[0])
    pred_props = K.eval(pred_props)[0]
    print(pred_props)
    z = np.array(K.eval(z[0])[0])
    print(z_to_y(z))

start_time = time.time()
para_path = "../parameters/predictor-20210525/"
model_path = "../models/viz/20210525/"
encoder = load_model(model_path + "encoder_best.h5")
decoder = load_model(model_path + "decoder_best.h5", custom_objects={'TerminalGRU':TerminalGRU})
predictor = load_model(model_path + "prop_pred_best.h5")

params = self_hyperparameters.load_params("modify_smiles")
print(params)

MAX_LEN = params['max_len']
CHAR_INDICES = json.load(open(params["dict_path"]))
print(CHAR_INDICES)
NCHARS = len(CHAR_INDICES)
params['NCHARS'] = NCHARS
INDICES_CHAR = dict((CHAR_INDICES[i], i) for i in CHAR_INDICES)
# print(CHAR_INDICES, INDICES_CHAR)


smile_list = pd.read_csv("../activity_coefficient_v1_gamma.csv")["canon_smiles"].tolist()


""" Latent variable normalization """
Zs = mean_var(model_path + "Zs.npy")
min, max = np.min(Zs, axis=0), np.max(Zs, axis=0)
scale = max - min
Zs = (Zs-min)/scale




# predictor - layer 1 (w & b & sigmoid)
w1 = np.load(para_path + "0@dense_4+kernel+0.npy").T
b1 = np.load(para_path + "1@dense_4+bias+0.npy").T
# predictor - layer 2 (w & b & softplus & batch_norm)
w2 = np.load(para_path + "2@property_predictor_dense0+kernel+0.npy").T
b2 = np.load(para_path + "3@property_predictor_dense0+bias+0.npy").T

# mu2 = np.load(para_path + "8@batch_normalization_1+moving_mean+0.npy")
# std2 = np.load(para_path + "9@batch_normalization_1+moving_variance+0.npy")
# gamma2 = np.load(para_path + "4@batch_normalization_1+gamma+0.npy")
# beta2 = np.load(para_path + "5@batch_normalization_1+beta+0.npy")
# predictor - layer 3 (w & b)
w3 = np.load(para_path + "4@reg_property_output+kernel+0.npy").T
b3 = np.load(para_path + "5@reg_property_output+bias+0.npy").T

# print(smiles_to_y("O=[N+]([O-])OCC(CO[N+](=O)[O-])O[N+](=O)[O-]"))


with open("./generate_solvent_smiles.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Target", "Gamma1", "Gamma2", "LatentVariable", "DeSMILES", "DeVAEtarget", "DeVAEgamma1", "DeVAEgamma2",
                     "CaSMILES", "CaVAEtarget", "CaVAEgamma1", "CaVAEgamma2"])

convergence = False
tolerance = 0.1
target_decay = 0.01
sol_target_ini = np.inf
while (not convergence) and (sol_target_ini >= 4.):
    Valid = False
    """ MODEL DEFINATION """
    model = ConcreteModel()
    # T = model.T = Var(domain=NonNegativeReals, bounds=(273.15, 373.15))

    """ LIQUID PHASE - ACTIVITY COEFFICIENT """
    model.latent_ind = range(128)
    model.z = Var(model.latent_ind, domain=NonNegativeReals, bounds=(0.0, 1.0))
    z = [model.z[i] * scale[i] + min[i] for i in model.latent_ind]

    # predictor - layer 1 (w & b & sigmoid)
    # w1 = np.load(para_path + "0@dense_4+kernel+0.npy").T
    # b1 = np.load(para_path + "1@dense_4+bias+0.npy").T
    # print(w1.shape, b1.shape)
    z = list(sum(z[i] * w[i] for i in model.latent_ind) for w in w1)
    z = [a + b for a, b in zip(z, b1)]
    # x = [tanh(i) for i in x]
    z = [sigmoid(i) for i in z]

    # predictor - layer 2 (w & b & softplus & batch_norm)
    # w2 = np.load(para_path + "2@property_predictor_dense0+kernel+0.npy").T
    # b2 = np.load(para_path + "3@property_predictor_dense0+bias+0.npy").T
    # print(w2.shape, b2.shape)
    z = list(sum(z[i] * w[i] for i in range(len(z))) for w in w2)
    z = [a + b for a, b in zip(z, b2)]
    # x = [tanh(i) for i in x]
    z = [softplus(i) for i in z]

    # mu2 = np.load(para_path + "8@batch_normalization_4+moving_mean+0.npy")
    # std2 = np.load(para_path + "9@batch_normalization_4+moving_variance+0.npy")
    # gamma2 = np.load(para_path + "4@batch_normalization_4+gamma+0.npy")
    # beta2 = np.load(para_path + "5@batch_normalization_4+beta+0.npy")
    # print(mu2.shape, std2.shape, gamma2.shape, beta2.shape)
    # z = (z - mu2) / (np.sqrt(std2 + 0.001)) * gamma2 + beta2

    # predictor - layer 3 (w & b)
    # w3 = np.load(para_path + "6@reg_property_output+kernel+0.npy").T
    # b3 = np.load(para_path + "7@reg_property_output+bias+0.npy").T
    # print(w3.shape, b3.shape)
    z = list(sum(z[i] * w[i] for i in range(len(z))) for w in w3)
    model.ln_gamma = np.array([a + b for a, b in zip(z, b3)])
    model.ln_ga_cons1 = Constraint(expr = (-1.46124589, model.ln_gamma[0], 3.88683326))
    model.ln_ga_cons2 = Constraint(expr = (-1.09950353, model.ln_gamma[1], 4.94676317))
    model.gamma = np.array([exp(model.ln_gamma[i]) for i in range(len(model.ln_gamma))])
    Gamma1, Gamma2 = model.gamma[0], model.gamma[1]

    sol_target = (Gamma2 / Gamma1) / Gamma1
    model.profit = Objective(expr = sol_target, sense=maximize)

    model.select = Constraint(expr= (Gamma2 / Gamma1) >= 1)
    model.capaci = Constraint(expr= (1 / Gamma1) >= 1)
    model.limit = Constraint(expr = sol_target <= sol_target_ini)
    # model.chemical_space = Constraint(expr = sum(model.z[i]**2 for i in model.latent_ind) <= 1)
    SolverFactory('ipopt', executable="C:/zwang2020Doc/coin_or/ipopt/ipopt.exe" ).solve(model)#.write()
    print('\nProfit = ', model.profit())
    sol_target_ini = model.profit()
    # print("Activity coefficient:", model.gamma[0](), model.gamma[1]())
    latent_var = list(model.z[i]() * scale[i] + min[i] for i in model.latent_ind)
    # print(latent_var)
    # np.save("Latent_var.npy", latent_var)

    print("Reproduction:", z_to_y(np.array(latent_var)))

    # VAE - Decoder
    fake_in = np.zeros((1, 20, 45))
    z = [K.variable([latent_var]), K.variable(fake_in)]
    X_r = decoder(z)
    X_r = K.eval(X_r)
    re_smile, de_smile = mu.hot_to_smiles(X_r, INDICES_CHAR, params["string_type"])
    re_smile, de_smile = re_smile[0], de_smile[0]
    if "invalid" in re_smile:
        sol_target_ini -= target_decay
    else:
        Valid = True
    print("Decoded SMILES:", de_smile, "/", "After canonlization:", re_smile)

    if Valid:
        if re_smile not in smile_list:
            # DeSMILES
            string = [modify_smiles(de_smile)]
            print(string)
            X = mu.string_to_hot(string, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
            z = encoder(tf.constant(X, dtype=tf.float32))
            pred_props = predictor(z[0])
            pred_props = K.eval(pred_props)[0]
            DeGamma1, DeGamma2 = exp(pred_props[0]), exp(pred_props[1])
            print("DeSMILES property:", pred_props, [DeGamma1, DeGamma2])

            # CaSMILES
            if "[SH]" in re_smile:
                re_smile = re_smile.replace("[SH]", "S")
            string = [modify_smiles(re_smile)]
            print(string)
            X = mu.string_to_hot(string, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
            z = encoder(tf.constant(X, dtype=tf.float32))
            pred_props = predictor(z[0])
            pred_props = K.eval(pred_props)[0]
            CaGamma1, CaGamma2 = exp(pred_props[0]), exp(pred_props[1])
            print("CaSMILES property:", pred_props, [CaGamma1, CaGamma2])

            update = [model.profit(), model.gamma[0](), model.gamma[1](), latent_var,
                      de_smile, DeGamma1, DeGamma2, DeGamma2 / DeGamma1 ** 2,
                      re_smile, CaGamma1, CaGamma2, CaGamma2 / CaGamma1 ** 2]
            print("Update:", update)
            with open("./generate_solvent_smiles.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(update)

            sol_target_ini -= target_decay
        else:
            print("old solvent")
            sol_target_ini -= target_decay


print("** FINISHED ** @time of run:", time.time() - start_time)
