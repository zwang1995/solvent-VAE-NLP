import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import yaml, json
import csv
import pandas as pd

pd.set_option("display.max_columns", None)
import numpy as np

# import tensorflow as tf
# import keras
# from keras import backend as K

# K.set_learning_phase(0)
# from models import load_encoder, load_decoder, load_property_predictor
# from keras.models import load_model
# from tgru_k2_gpu import TerminalGRU
import hyperparameters
import mol_utils as mu
from zw_smiles_split import modify_smiles, inverse_modify_smiles

# import selfies, deepsmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from sklearn.externals import joblib
from rdkit import Chem


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def cross_entropy(target, output):
    output /= tf.reduce_sum(output,
                            reduction_indices=len(output.get_shape()) - 1,
                            keep_dims=True)
    # manual computation of crossentropy
    epsilon = 1e-7
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return K.eval(- tf.reduce_sum(target * tf.log(output),
                                  reduction_indices=len(output.get_shape()) - 1))


def category_accuracy(target, output):
    accu = keras.metrics.categorical_accuracy(target, output)
    accu = K.eval(accu)
    return np.sum(accu) / len(accu)


plt.rcParams["font.size"] = "11"
plt.rcParams["font.family"] = "arial"
plt.rcParams["figure.figsize"] = (4, 3)

s = 100
alpha = 0.8


def data_generation(model_num):
    """ Load parameters """
    params = hyperparameters.load_params("modify_smiles")

    MAX_LEN = params["max_len"]
    CHAR_INDICES = json.load(open(params["dict_path"]))
    NCHARS = len(CHAR_INDICES)
    params["NCHARS"] = NCHARS
    INDICES_CHAR = dict((CHAR_INDICES[i], i) for i in CHAR_INDICES)
    print(CHAR_INDICES, INDICES_CHAR)

    # load models
    encoder = load_model("../models/viz/model_for_viz/encoder_" + str(model_num) + ".h5")
    decoder = load_model("../models/viz/model_for_viz/decoder_" + str(model_num) + ".h5",
                         custom_objects={"TerminalGRU": TerminalGRU})
    if params["do_prop_pred"]:
        predictor = load_model("../models/viz/model_for_viz/prop_pred_" + str(model_num) + ".h5")

    # load data - normal case
    gamma1, gamma2 = "log_gamma_1", "log_gamma_2"
    prop_df = pd.read_csv("../activity_coefficient_v1_gamma.csv", usecols=["canon_smiles", gamma1, gamma2, "name"])
    prop_df["smiles"] = prop_df.canon_smiles.str.strip()

    if params["string_type"] == "modify_smiles":
        prop_df["modify_smiles"] = prop_df.smiles.apply(lambda x: modify_smiles(x))
        prop_df["string_len"] = prop_df.modify_smiles.apply(lambda x: len(x))
    # elif params["string_type"] == "selfies":
    #     prop_df["selfies"] = prop_df.smiles.apply(lambda x: selfies.encoder(x))
    #     prop_df["string_len"] = prop_df.selfies.apply(lambda x: selfies.len_selfies(x))
    # elif params["string_type"] == "deepsmiles":
    #     converter = deepsmiles.Converter(rings=True, branches=True)
    #     prop_df["deepsmiles"] = prop_df.smiles.apply(lambda x: converter.encode(x))
    #     prop_df["string_len"] = prop_df.deepsmiles.apply(lambda x: len(x))

    prop_df = prop_df[prop_df.string_len <= params["max_len"]]
    smiles = prop_df.smiles.tolist()

    if params["string_type"] == "modify_smiles":
        strings = prop_df.modify_smiles.tolist()
    # elif params["string_type"] == "selfies":
    #     strings = prop_df.selfies.tolist()
    # elif params["string_type"] == "deepsmiles":
    #     strings = prop_df.deepsmiles.tolist()

    names = prop_df["name"].tolist()
    gamma1 = prop_df[gamma1].tolist()
    gamma2 = prop_df[gamma2].tolist()
    string_lens = prop_df.string_len.tolist()
    total_string = len(strings)
    print("Total string:", total_string)

    X = mu.string_to_hot(strings, MAX_LEN, CHAR_INDICES, NCHARS, params["string_type"])
    z = encoder(tf.constant(X, dtype=tf.float32))
    z = [z[0], K.variable(X)]
    X_r = decoder(z)
    X_r = K.eval(X_r)
    if params["do_prop_pred"]:
        pred_props = predictor(z[0])
        pred_props = K.eval(pred_props)
    else:
        pred_props = [[0]] * total_string
    re_can_smiles, re_smiles = mu.hot_to_smiles(X_r, INDICES_CHAR, params["string_type"])
    print(re_can_smiles, re_smiles)
    Xs, X_rs, Zs = X, X_r, K.eval(z[0])
    np.save("../models/viz/Zs_Solvents.npy", np.array(Zs))

    data_path = "../models/viz/data.csv"
    with open(data_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "smiles", "resmiles", "recansmiles", "stringlen",
                         "gamma1", "gamma2", "pred_gamma1", "pred_gamma2", "error1", "error2"])

    # losses, accus = [], []
    # for (i, (X, X_r)) in enumerate(zip(Xs, X_rs)):
    #     print(str(i + 1))
    # losses.append(np.average(cross_entropy(K.variable(X), K.variable(X_r))))
    # accus.append(category_accuracy(K.variable(X), K.variable(X_r)))

    with open(data_path, "a", newline="") as f:
        writer = csv.writer(f)
        accus = []
        for (i, (name, smile, re_smile, re_can_smile, string_len, gamma_1, gamma_2, pred_prop)) in enumerate(
                zip(names, smiles, re_smiles, re_can_smiles, string_lens, gamma1, gamma2, pred_props)):
            print(str(i + 1), "/", total_string, ":", [gamma_1, gamma_2], pred_prop)
            # loss = np.average(cross_entropy(K.variable(X), K.variable(X_r)))
            cate_accu = category_accuracy(K.variable(X[i]), K.variable(X_r[i]))
            accus.append(cate_accu)
            writer.writerow([name, smile, re_smile, re_can_smile, string_len, gamma_1, gamma_2,
                             pred_prop[0], pred_prop[1], abs(gamma_1 - pred_prop[0]), abs(gamma_2 - pred_prop[1])])
    print(np.mean(accus))
    # df_out = pd.DataFrame(data, columns=["OriginalSMILES", "ReconSMILES", "StringLen", "Loss", "CategoricalAccuracy",
    #                                      "LatentVariable", "gamma1", "gamma2", "pred_gamma1", "pred_gamma2"])
    # df_out["PropError1"] = np.abs(np.array(df_out.gamma1.tolist()) - np.array(df_out.pred_gamma1.tolist()))
    # df_out["PropError2"] = np.abs(np.array(df_out.gamma2.tolist()) - np.array(df_out.pred_gamma2.tolist()))
    # df_out.to_csv("../models/viz/data.csv")
    # Zs = df_out.LatentVariable.tolist()
    # # print(Zs)


def reduce_dimension(df, Zs):
    # do pca and normalize
    pca = PCA(n_components=2)
    # print(Zs)
    Z_pca = pca.fit_transform(Zs)
    joblib.dump(pca, "../models/viz/pca.pkl")
    scaler = MinMaxScaler()
    Z_pca = scaler.fit_transform(Z_pca)
    joblib.dump(scaler, "../models/viz/pca_scaler.pkl")
    np.save("../models/viz/Zs_pca_Solvents.npy", np.array(Z_pca))

    df_pca = pd.DataFrame(np.transpose((Z_pca[:, 0], Z_pca[:, 1])))
    df_pca.columns = ["x", "y"]
    df_pca["name"] = df.name
    df_pca["gamma1"] = df.gamma1.tolist()
    df_pca["gamma2"] = df.gamma2.tolist()
    df_pca["pred_gamma1"] = df.pred_gamma1.tolist()
    df_pca["pred_gamma2"] = df.pred_gamma2.tolist()
    # df_pca["loss"] = df.Loss
    # df_pca["cate_accu"] = df.CategoricalAccuracy
    df_pca["error1"] = df.error1
    df_pca["error2"] = df.error2
    df_pca["len"] = df.stringlen
    df_pca["name"] = df.name
    df_pca.to_csv("../models/viz/pca_viz.csv")

    # # do tsne and normalize
    # tsne = TSNE(n_components=2)
    # Z_tsne = tsne.fit_transform(Zs)
    # joblib.dump(tsne, "../models/viz/tsne.pkl")
    # scaler = MinMaxScaler()
    # Z_tsne = scaler.fit_transform(Z_tsne)
    # joblib.dump(scaler, "../models/viz/tsne_scaler.pkl")
    # np.save("../models/viz/Z_tsne.npy", np.array(Z_tsne))
    #
    # df_tsne = pd.DataFrame(np.transpose((Z_tsne[:, 0], Z_tsne[:, 1])))
    # df_tsne.columns = ["x", "y"]
    # df_tsne["gamma1"] = df.gamma1.tolist()
    # df_tsne["gamma2"] = df.gamma2.tolist()
    # df_tsne["loss"] = df.Loss
    # df_tsne["cate_accu"] = df.CategoricalAccuracy
    # df_tsne["error1"] = df.PropError1
    # df_tsne["error2"] = df.PropError2
    # df_tsne["len"] = df.StringLen
    # df_tsne.to_csv("../models/viz/tsne_viz.csv")


def data_viz(df, comp_type="pca"):
    " Visualization "
    " Property distribution "
    if comp_type == "pca":
        x_label = "Principal component 1"
        y_label = "Principal component 2"
    elif comp_type == "tsne":
        x_label = "TSNE_1"
        y_label = "TSNE_2"

    def get_setting(prop):
        if prop == "gamma1":
            c = df.gamma1
            cmap = "jet"
            c_low, c_high = -2, 5
        if prop == "gamma2":
            c = df.gamma2
            cmap = "jet"
            c_low, c_high = -2, 5
        if prop == "pred_gamma1":
            c = df.pred_gamma1
            cmap = "jet"
            c_low, c_high = -2, 5
        if prop == "pred_gamma2":
            c = df.pred_gamma2
            cmap = "jet"
            c_low, c_high = -2, 5
        if prop == "error1":
            c = df.error1
            cmap = "viridis_r"
            c_low, c_high = 0, 1.2
        if prop == "error2":
            c = df.error2
            cmap = "viridis_r"
            c_low, c_high = 0, 1.2
        if prop == "cate_accu":
            c = df.cate_accu
            cmap = "jet_r"
            c_low, c_high = 0, 1
        return c, cmap, c_low, c_high

    for prop in ["gamma1", "gamma2", "pred_gamma1", "pred_gamma2", "error1", "error2", "cate_accu"]:
        plt.clf()
        c, cmap, c_low, c_high = get_setting(prop)
        plt.scatter(x=df["x"], y=df["y"], c=c,
                    cmap=cmap, marker=".",
                    s=s, alpha=alpha, edgecolors="none")
        plt.xlabel(x_label, size=12)
        plt.ylabel(y_label, size=12)
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.tick_params(axis="x", direction="in")
        plt.tick_params(axis="y", direction="in")
        cbar = plt.colorbar(pad=0.03)
        cbar.ax.tick_params(labelsize=10)
        plt.clim(c_low, c_high)
        plt.savefig("".join(["../models/viz/", comp_type, "_", prop]), dpi=300, bbox_inches="tight", transparent=True)


        # " Model performance "
        # cmap = plt.get_cmap("viridis")
        # new_cmap = truncate_colormap(cmap, 0, 0.8)

        # plt.clf()
        # plt.scatter(x=df["x"], y=df["y"], c=df.loss,
        #             cmap="plasma", marker=".",
        #             s=s, alpha=alpha, edgecolors="none")
        # plt.xlabel(x_label, size=16)
        # plt.ylabel(y_label, size=16)
        # # plt.title("Reconstruction Loss", size=14)
        # plt.colorbar()
        # plt.savefig("../models/viz/" + comp_type + "_3_recon_loss.tif", dpi=300)


    plt.clf()
    x, y, z = df["x"], df["y"], df.len
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x=x, y=y, c=z,
                cmap="jet_r", marker=".",
                s=s, alpha=alpha, edgecolors="none")
    plt.xlabel(x_label, size=12)
    plt.ylabel(y_label, size=12)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    # plt.title("String length")
    cbar = plt.colorbar(pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    plt.clim(0, 20)
    plt.savefig("../models/viz/" + comp_type + "_len", dpi=300, bbox_inches="tight", transparent=True)


def molecule_location(Z, df):
    """"""
    """ Property prediction """
    # predictor = load_model("../models/viz/model_for_viz/prop_pred_" + str(model_num) + ".h5")
    # pred_props = predictor(K.variable(Z))
    # pred_props = K.eval(pred_props)[0][0]
    # print("Predicted property:", pred_props)

    """ Molecule location """
    Z0 = [np.load("../models/viz/Zs_Solvents.npy")[123, :]]
    pca_model = joblib.load("../models/viz/pca.pkl")
    Z_pca = pca_model.transform(Z)
    Z0_pca = pca_model.transform(Z0)
    scaler = joblib.load("../models/viz/pca_scaler.pkl")
    Z = scaler.transform(Z_pca)
    Z0 = scaler.transform(Z0_pca)
    print(Z, Z0)

    mappable = plt.scatter(x=df["x"], y=df["y"], c=(df.gamma1), edgecolors="seagreen", linewidth=1, alpha=0.5, s=20)
    plt.clf()
    plt.scatter(x=df["x"], y=df["y"], c="silver",  # c=(df.gamma1), cmap="jet",
                marker=".",
                s=s, alpha=alpha, edgecolors="none")
    plt.xlabel("Principal component 1", size=13)
    plt.ylabel("Principal component 2", size=13)
    plt.colorbar(mappable)
    for i in range(len(Z)):
        plt.scatter(x=Z[4 - i, 0], y=Z[4 - i, 1], c="r", marker="X", edgecolors="k", linewidth=1, alpha=0.7, s=s,
                    zorder=5 * i)

    " case 1 "
    # c, lw = "lime", 2
    # plt.plot([0.49, 0.49], [0.44, 0.66], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.49, 0.71], [0.44, 0.44], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.71, 0.71], [0.44, 0.66], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.49, 0.71], [0.66, 0.66], c=c, linewidth=lw, linestyle="--")
    #
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.savefig("../models/viz/0_new_solvent", dpi=300, bbox_inches="tight", transparent=True)

    " case 2 "
    # c, lw = "lime", 2
    # plt.plot([0.558, 0.558], [0.508, 0.562], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.558, 0.612], [0.508, 0.508], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.612, 0.612], [0.508, 0.562], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.558, 0.612], [0.562, 0.562], c=c, linewidth=lw, linestyle="--")
    #
    # plt.xlim([0.49,0.71])
    # plt.ylim([0.44,0.66])
    # plt.xticks([0.5, 0.55, 0.6, 0.65, 0.7])
    # plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65])
    # plt.savefig("../models/viz/0_new_solvent_small", dpi=300, bbox_inches="tight", transparent=True)

    " case 3 "
    # plt.xlim([0.558,0.612])
    # plt.ylim([0.508,0.562])
    # plt.xticks([0.56, 0.57, 0.58, 0.59, 0.60, 0.61])
    # # plt.yticks([0.4, 0.5, 0.6, 0.7])
    # plt.savefig("../models/viz/0_new_solvent_tiny", dpi=300, bbox_inches="tight", transparent=True)


def sampling_between(Z, df):
    """"""
    """ Molecule location """
    Z0 = [np.load("../models/viz/Zs_Solvents.npy")[123, :]]
    pca_model = joblib.load("../models/viz/pca.pkl")
    Z_pca = pca_model.transform(Z)
    Z0_pca = pca_model.transform(Z0)
    scaler = joblib.load("../models/viz/pca_scaler.pkl")
    Z = scaler.transform(Z_pca)
    Z0 = scaler.transform(Z0_pca)
    print(Z, Z0)

    mappable = plt.scatter(x=df["x"], y=df["y"], c=(df.gamma1), edgecolors="seagreen", linewidth=1, alpha=0.5, s=20)
    plt.clf()
    plt.scatter(x=df["x"], y=df["y"], c="silver", marker=".", s=s, alpha=alpha, edgecolors="none", zorder=0)
    plt.xlabel("Principal component 1", size=12)
    plt.ylabel("Principal component 2", size=12)
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")
    plt.colorbar(mappable)
    plt.scatter(x=Z0[0][0], y=Z0[0][1], c="b", marker="*", edgecolors="k", linewidth=1, alpha=1, s=1.5 * s, zorder=2)

    Zs = np.array([[0.581345602, 0.613322588],
                   [0.583478622, 0.597356196],
                   # [0.586038246, 0.578196525],
                   [0.586251548, 0.576599886],
                   [0.587104756, 0.570213329],
                   # [0.587211407, 0.56941501],
                   # [0.587424708, 0.567818371],
                   [0.587744661, 0.565423412],
                   # [0.587851312, 0.564625092],
                   # [0.588064614, 0.563028453],
                   # [0.589664379, 0.551053659],
                   [0.589877681, 0.54945702],
                   # [0.591050842, 0.540675504],
                   [0.591264144, 0.539078865],
                   # [0.591477446, 0.537482226],
                   [0.591690748, 0.535885587]])
    plt.scatter(x=Z[0][0], y=Z[0][1], c="r", marker="*", edgecolors="k", linewidth=1, alpha=1, s=1.5 * s, zorder=3)

    " case 1 "
    # c, lw = "lime", 2
    # plt.plot([0.538, 0.538], [0.528, 0.622], c=c, linewidth=lw, linestyle="-", zorder=1)
    # plt.plot([0.538, 0.632], [0.528, 0.528], c=c, linewidth=lw, linestyle="-", zorder=1)
    # plt.plot([0.632, 0.632], [0.528, 0.622], c=c, linewidth=lw, linestyle="-", zorder=1)
    # plt.plot([0.538, 0.632], [0.622, 0.622], c=c, linewidth=lw, linestyle="-", zorder=1)
    #
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.tick_params(axis="x", direction="in")
    # plt.tick_params(axis="y", direction="in")
    # plt.savefig("../models/viz/0_sampling_between", dpi=300, bbox_inches="tight", transparent=True)

    " case 2 "
    # c, lw = "lime", 2
    # plt.plot([0.558, 0.558], [0.508, 0.562], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.558, 0.612], [0.508, 0.508], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.612, 0.612], [0.508, 0.562], c=c, linewidth=lw, linestyle="--")
    # plt.plot([0.558, 0.612], [0.562, 0.562], c=c, linewidth=lw, linestyle="--")
    #
    # plt.xlim([0.49,0.66])
    # plt.ylim([0.49,0.66])
    # plt.xticks([0.5, 0.55, 0.6, 0.65])
    # plt.yticks([0.5, 0.55, 0.6, 0.65])
    # plt.savefig("../models/viz/0_sampling_between_small", dpi=300, bbox_inches="tight")

    " case 3 "
    # plt.scatter(x=Zs[:,0], y=Zs[:,1], c="g", marker="X", edgecolors="k", linewidth=1, alpha=0.7, s=s, zorder=2)
    # plt.xlim([0.538, 0.632])
    # plt.ylim([0.528, 0.622])
    # plt.xticks([0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63])
    # plt.yticks([0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62])
    # plt.savefig("../models/viz/0_sampling_between_tiny", dpi=300, bbox_inches="tight", transparent=True)


def Zs_recovery(x, comp_type=None):
    if comp_type == "pca":
        Z_pca_scaler = joblib.load("../models/viz/pca_scaler.pkl")
        Z_pca = joblib.load("../models/viz/pca.pkl")
        x = Z_pca_scaler.inverse_transform(x)
        Zs = Z_pca.inverse_transform(x)
        return Zs

    # elif comp_type == "tsne":
    #     Z_tsne_scaler = joblib.load("../models/viz/tsne_scaler.pkl")
    #     Z_tsne = joblib.load("../models/viz/tsne.pkl")
    #     x = Z_tsne_scaler.inverse_transform(x)
    #     Zs = Z_tsne.inverse_transform(x)
    #     return Zs


if __name__ == "__main__":
    model_num = "best"

    """ Step 1: load models and molecules to calculate latent variables, losses, etc. """
    # data_generation(model_num)

    """ Step 2: reduce dimension of Zs into 2 """
    # reduce_dimension(pd.read_csv("../models/viz/data.csv"), np.load("../models/viz/Zs_Solvents.npy"))

    """ Step 3: visualization """
    df_pca = pd.read_csv("../models/viz/pca_viz.csv")
    # data_viz(df_pca, "pca")

    """ Step 4: new molecule """
    Z = np.load("Zs_CandidateTop5.npy")
    # molecule_location(Z, df_pca)

    """ Step 5: Sampling between """
    # Z = np.load("Zs_CandidateTop1.npy")
    sampling_between(Z, df_pca)

    # Zs_recovery(np.load("../models/viz/Zs_pca_Solvents.npy"), "pca")
