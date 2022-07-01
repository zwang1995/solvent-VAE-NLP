from models import load_encoder, load_decoder, load_property_predictor
import hyperparameters
import mol_utils as mu
from zw_smiles_split import modify_smiles
import yaml, json
import tensorflow as tf
from keras import backend as K
K.set_learning_phase(0)

import numpy as np
from keras.models import load_model
from tgru_k2_gpu import TerminalGRU

""" Load parameters """
params = hyperparameters.load_params("modify_smiles")

MAX_LEN = params['MAX_LEN']
PADDING = params['PADDING']

CHAR_INDICES = json.load(open(params["dict_path"]))
NCHARS = len(CHAR_INDICES)
params['NCHARS'] = NCHARS
INDICES_CHAR = dict((CHAR_INDICES[i], i) for i in CHAR_INDICES)
print(CHAR_INDICES, INDICES_CHAR)

smile = ['O=[N+]([O-])OCC(CO[N+](=O)[O-])O[N+](=O)[O-]']
smile = [modify_smiles(i) for i in smile]
print(smile)
X = mu.string_to_hot(smile, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
X = X.reshape(-1, MAX_LEN, NCHARS)


model_num = "best"
model_path = "../models/viz/20210623/"
encoder = load_model(model_path + "encoder_" + str(model_num) + ".h5")
decoder = load_model(model_path + "decoder_" + str(model_num) + ".h5", custom_objects={'TerminalGRU': TerminalGRU})
if params["do_prop_pred"]:
    predictor = load_model(model_path + "prop_pred_" + str(model_num) + ".h5")

z = encoder(tf.constant(X, dtype=tf.float32))
z = [z[0], K.variable(X)]
# Z = K.eval(z[0])
# print(Z)
X_r = decoder(z)
X_r = K.eval(X_r)
re_smile, de_smiles = mu.hot_to_smiles(X_r, INDICES_CHAR, params["string_type"])
print(re_smile, de_smiles)
pred_props = predictor(z[0])
pred_props = K.eval(pred_props)[0]
print(pred_props)
print(np.exp(pred_props[0]), np.exp(pred_props[1]))