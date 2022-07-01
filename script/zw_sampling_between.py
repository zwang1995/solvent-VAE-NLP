import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import pandas as pd

import hyperparameters
import mol_utils as mu
from tgru_k2_gpu import TerminalGRU
from zw_smiles_split import modify_smiles, inverse_modify_smiles

import tensorflow as tf
from keras.models import load_model
from keras import backend as K

K.set_learning_phase(0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.externals import joblib

params = hyperparameters.load_params("modify_smiles")

MAX_LEN = params['max_len']
CHAR_INDICES = json.load(open(params["dict_path"]))
NCHARS = len(CHAR_INDICES)
params['NCHARS'] = NCHARS
INDICES_CHAR = dict((CHAR_INDICES[i], i) for i in CHAR_INDICES)
print(CHAR_INDICES, INDICES_CHAR)

encoder = load_model("../models/viz/20210525/encoder_best.h5")
decoder = load_model("../models/viz/20210525/decoder_best.h5", custom_objects={'TerminalGRU': TerminalGRU})
predictor = load_model("../models/viz/20210525/prop_pred_best.h5")

smile = mu.canon_smiles('O=[N+]([O-])OCC(CO[N+](=O)[O-])O[N+](=O)[O-]')
string = [modify_smiles(smile)]
X = mu.string_to_hot(string, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
z = encoder(tf.constant(X, dtype=tf.float32))

Z1 = [0.2671867158375344, -0.14021305440746012, 0.004325031044225658, -0.1925177999505856, 0.21879887193725756,
      0.443686013914001, 0.004425627619982664, 0.09043223708688564, 0.11266506893240313, 0.11509413796737267,
      0.020330791611747556, -0.03930554812480508, -0.15261667326233436, -0.025104581261338632, -0.1860287496895523,
      0.13729707861167306, -0.12944268689979932, -0.1570858176102814, -0.005466622325793424, 0.20083650045285223,
      -0.25536390399800946, -0.25288866633449364, 0.4997922563393633, 0.10072479648689991, 0.3751604699908577,
      -0.31450885867057554, 0.4098206276700036, 0.1888891055630464, 0.14829166208730937, 0.2846357060224982,
      -0.34011865501205807, 0.14067305737478275, 0.18489544729996643, 0.2222851419448324, 0.24823101516200152,
      -0.0030351218359926935, 0.18862744356503547, 0.03332973103656994, 0.11281143260501025, 0.2241817907307566,
      0.20206289579248438, 0.021125344756774578, -0.18207626897389134, -0.38106305991888034, 0.3207979801989148,
      0.19847284577946123, 0.11843934833338643, -0.17105057912027505, -0.04131459331844367, 0.28884734744583485,
      0.14392853875187944, -0.2461819330333272, 0.20753854494063317, 0.1898784565500624, 0.1183987686311892,
      -0.0852066869363039, 0.19107881349524092, 0.1836090148010704, 0.057398056404746045, -0.33019660646073934,
      -0.33626406627649597, -0.38634144768809076, -0.5637289612557161, 0.2781962086716775, 0.2738952442479947,
      0.3702487934444235, -0.15032801833892212, 0.060600748977777785, 0.012080096095114345, 0.05907917943348995,
      -0.2644361177179825, -0.011658554283973777, -0.35496338419054013, 0.14683567467517156, -0.03082028458562386,
      -0.25478510234214247, 0.0636513730399847, 0.3447284054048996, 0.32450104192236295, -0.06882533041724992,
      -0.15433385464746296, 0.24060474014510214, -0.039912518919530515, -0.45152559853291635, -0.13806645607553802,
      -0.1294385864962307, -0.15537536179047834, -0.4782300611584576, 0.006087187017407458, 0.8176218563583275,
      -0.380841307168971, 0.42356437585900153, 0.18486895010466653, 0.11753213522684586, 0.2936556298370723,
      0.03271347689900089, 0.15713593335281129, 0.03484576161909608, -0.014020522562603333, 0.31595360710443166,
      0.3387735706459567, -0.2179052076687258, 0.14313034653196766, -0.049537409543973954, -0.18709935828051094,
      -0.25973004849104364, 0.3526413391733879, 0.12524333468863713, 0.32594477263378185, -0.008237592441602659,
      -0.059593308404811585, 0.22532119268443118, -0.09715651126576857, 0.27234924515472503, -0.2663795865889642,
      -0.401901493271547, 0.25613447084470187, -0.09880581006315414, -0.14836918277100858, -0.06350699266165444,
      -0.00678864204950641, 0.3808869623449358, -0.18254825344197212, -0.288584906404035, 0.1814724049139237,
      -0.1410074363357276, -0.1854008639922976, -0.19721523295850824]

attempts = 100
# noise = 100

Z0 = K.eval(z[0])[0]
Z = np.tile(Z0, (attempts + 1, 1))  # shape(100,128)
delta = (Z1 - Z0) / attempts
DELTA = np.tile(delta, (attempts + 1, 1))
i = np.diag(np.arange(attempts+1)[:])
Zs = np.dot(i, DELTA) + Z


fake_shape = (Zs.shape[0], X.shape[1], X.shape[2])
fake_in = np.zeros(fake_shape)

X = decoder([K.variable(Zs), K.variable(fake_in)])
X = K.eval(X)
smiles = mu.hot_to_smiles(X, INDICES_CHAR, params["string_type"])[0]
pred_props = predictor(K.variable(Zs))
pred_props = K.eval(pred_props)
print(pred_props)


def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()
        index += 1
    return balanced and len(s) == 0


def matched_ring(s):
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0


def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)


def smiles_distance_z(smiles, z0):
    string = [modify_smiles(smiles)]
    x = mu.string_to_hot(string, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])

    z_rep = K.eval(encoder(tf.constant(x, dtype=tf.float32))[0])
    return np.linalg.norm(z0 - z_rep, axis=1)

def distance_z(z, z0):
    return np.linalg.norm(z0 - z)

def prop_pred(smiles):
    string = [modify_smiles(smiles)]
    x = mu.string_to_hot(string, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
    z_rep = K.eval(encoder(tf.constant(x, dtype=tf.float32))[0])
    return K.eval(predictor(K.variable(z_rep)))[0]

dis = [distance_z(z, Z0) for z in Zs]

pca_model = joblib.load("../models/viz/pca.pkl")
Zs_pca = pca_model.transform(Zs)
scaler = joblib.load("../models/viz/pca_scaler.pkl")
Zs = scaler.transform(Zs_pca)
X, Y = Zs[:,0], Zs[:,1]

df = pd.DataFrame({"smiles": smiles, "X": X, "Y": Y})
# sort_df = pd.DataFrame(df[["smiles"]].groupby(by="smiles").size().rename("count").reset_index())
# df = df.merge(sort_df, on="smiles")
# df.drop_duplicates(subset="smiles", inplace=True)
# df = df[df['smiles'].apply(fast_verify)]
# if len(df) > 0:
#     df['mol'] = df['smiles'].apply(mu.smiles_to_mol)
# if len(df) > 0:
#     df = df[pd.notnull(df['mol'])]
# if len(df) > 0:
#     df['distance'] = df['smiles'].apply(lambda x: smiles_distance_z(x, Z0))
    # df['frequency'] = df['count'] / float(sum(df['count']))
    # df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
    # df.sort_values(by='distance', inplace=True)
    # df.reset_index(drop=True, inplace=True)
df['distance'] = dis
df["gamma1"] = pred_props[:,0]
df["gamma2"] = pred_props[:,1]
df["target"] = np.exp(df.gamma2 - 2 * df.gamma1)
# df = df.drop(columns=["mol"])
print(df)
df.to_csv("../sampling_between.csv")
