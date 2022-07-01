import pandas as pd
import numpy as np
import pickle as pkl
from rdkit.Chem import AllChem as Chem
import logging
# import selfies, deepsmiles

logging.getLogger("autoencoder")
logging.getLogger().setLevel(20)
logging.getLogger().addHandler(logging.StreamHandler())

from zw_smiles_split import modify_smiles, inverse_modify_smiles
# from zw_deepsmiles_split import modify_deepsmiles


# =================
# text io functions
# ==================

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None


def verify_smiles(smile):
    return (smile != "") and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)


def good_smiles(smile):
    if verify_smiles(smile):
        return canon_smiles(smile)
    else:
        return None


def pad_smile(string, max_len, padding="right"):
    if len(string) <= max_len:
        if padding == "right":
            return string + " " * (max_len - len(string))
        elif padding == "left":
            return " " * (max_len - len(string)) + string
        elif padding == "none":
            return string


def filter_valid_length(strings, max_len):
    return [s for s in strings if len(s) <= max_len]


def filter_valid_smiles_return_invalid(strings, max_len):
    filter_list = []
    new_smiles = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
    return new_smiles, filter_list


def string_to_hot(strings, max_len, char_indices, nchars, string_type):
    # smiles = [pad_smile(i, max_len, padding)
    #           for i in smiles if pad_smile(i, max_len, padding)]
    # extent the strings to a length of 120 with blanks

    X = np.zeros((len(strings), max_len, nchars), dtype=np.float32)
    # shape(5000, 120, 35)

    for i, string in enumerate(strings):
        if string_type == "smiles":
            string_len = len(string)
            string_list = list(string)
        # elif string_type == "selfies":
        #     string_len = selfies.len_selfies(string)
        #     string_list = list(selfies.split_selfies(string))
        # elif string_type == "deepsmiles":
        #     string_len = len(string)
        #     string_list = list(string)
        # elif string_type == "modify_deepsmiles":
        #     string_len = len(string)
        #     string_list = list(string)
        elif string_type == "modify_smiles":
            string_len = len(string)
            string_list = list(string)
        for t, char in enumerate(string_list):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", string)
                raise e
        X[i, string_len:, char_indices[" "]] = 1
        if np.sum(X[i]) != max_len:
            print("Error in one hot matrix")
    # print(X[0])
    # np.save("../parameters/smiles.npy", X[0])
    return X

def string_to_hot_DA(index, or_smiles, smiles, strings, y, max_len, char_indices, nchars, string_type):
    # smiles = [pad_smile(i, max_len, padding)
    #           for i in smiles if pad_smile(i, max_len, padding)]
    # extent the strings to a length of 120 with blanks

    X, Target, new_smiles, new_or_smiles, new_index = [], [], [], [], []
    # X = np.zeros((len(x), max_len, nchars), dtype=np.float32)
    # shape(5000, 120, 35)

    for string, target, smile, or_smile, i in zip(strings, y, smiles, or_smiles, index):
        Xi = np.zeros((max_len, nchars), dtype=np.float32)

        if string_type == "smiles":
            string_len = len(string)
            string_list = list(string)
        # elif string_type == "selfies":
        #     string_len = selfies.len_selfies(string)
        #     string_list = list(selfies.split_selfies(string))
        # elif string_type == "deepsmiles":
        #     string_len = len(string)
        #     string_list = list(string)
        # elif string_type == "modify_deepsmiles":
        #     string_len = len(string)
        #     string_list = list(string)
        elif string_type == "modify_smiles":
            string_len = len(string)
            string_list = list(string)

        try:
            for t, char in enumerate(string_list):
                Xi[t, char_indices[char]] = 1
            Xi[string_len:, char_indices[" "]] = 1

            if np.sum(Xi) != max_len:
                print("Error in one hot matrix")
            X.append(Xi)
            Target.append(target)
            new_smiles.append(smile)
            new_or_smiles.append(or_smile)
            new_index.append(i)
        except:
            pass
    # print(X[0])
    # np.save("../parameters/smiles.npy", X[0])
    X = np.array(X, dtype=np.float32)
    Target = np.array(Target, dtype=np.float32)
    return new_index, new_or_smiles, new_smiles, X, Target



def smiles_to_hot_filter(smiles, char_indices):
    filtered_smiles = []
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
    return filtered_smiles


def term_hot_to_smiles(x, temperature, indices_chars):
    temp_string = ""
    for j in x:
        index = thermal_argmax(j, temperature)
        temp_string += indices_chars[index]
    return temp_string


def hot_to_smiles(hot_x, indices_chars, string_type):
    smiles, strs = [], []
    invalid = 0
    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        if string_type == "smiles":
            str = temp_str.strip()
            strs.append(str)
            if Chem.MolFromSmiles(str) is None:
                smiles.append(str + "_invalid")
                invalid += 1
            else:
                smiles.append(Chem.CanonSmiles(str))
        elif string_type == "modify_smiles":
            str = temp_str.strip()
            str = inverse_modify_smiles(str)
            strs.append(str)
            if Chem.MolFromSmiles(str) is None:
                smiles.append(str + "_invalid")
                invalid += 1
            else:
                print(str, charge_N(str))
                if charge_N(str):
                    smiles.append(Chem.CanonSmiles(str))
                else:
                    smiles.append(str + "_invalid")
                    invalid += 1
        # elif string_type == "selfies":
        #     str = temp_str.strip()
        #     strs.append(str)
        #     self_str = selfies.decoder(str)
        #     if self_str is None:
        #         smiles.append(str + "_invalid")
        #         invalid += 1
        #     else:
        #         # str = self_str # for NLP
        #         if Chem.MolFromSmiles(self_str) is None:
        #             smiles.append(str + "_invalid")
        #             invalid += 1
        #         else:
        #             smiles.append(Chem.CanonSmiles(self_str))
        # elif (string_type == "deepsmiles") or (string_type == "modify_deepsmiles"):
        #     converter = deepsmiles.Converter(rings=True, branches=True)
        #     try:
        #         smile = converter.decode(temp_str.strip())
        #         if Chem.MolFromSmiles(smile) is None:
        #             smiles.append(smile + "_invalid2")
        #             invalid += 1
        #         else:
        #             smiles.append(Chem.CanonSmiles(smile))
        #     except:
        #         invalid += 1
        #         smiles.append(temp_str.strip()+"_invalid1")
    print("Invalid SMILES: {} / {}, Validity: {:2.2%}".format(invalid, len(hot_x), 1-invalid/len(hot_x)))
    return smiles, strs

def hot_to_strings_without_validation(hot_x, indices_chars, string_type):
    smiles = []
    invalid = 0
    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        if string_type == "smiles":
            str = temp_str.strip()
            smiles.append(str)
        elif string_type == "modify_smiles":
            str = temp_str.strip()
            str = inverse_modify_smiles(str)
            smiles.append(str)
        # elif string_type == "selfies":
        #     str = temp_str.strip()
        #     self_str = selfies.decoder(str)
        #     if self_str is None:
        #         smiles.append(str + "_invalid")
        #         invalid += 1
        #     else:
        #         if Chem.MolFromSmiles(self_str) is None:
        #             smiles.append(str + "_invalid")
        #             invalid += 1
        #         else:
        #             smiles.append(Chem.CanonSmiles(self_str))
        # elif (string_type == "deepsmiles") or (string_type == "modify_deepsmiles"):
        #     converter = deepsmiles.Converter(rings=True, branches=True)
        #     try:
        #         smile = converter.decode(temp_str.strip())
        #         if Chem.MolFromSmiles(smile) is None:
        #             smiles.append(smile + "_invalid2")
        #             invalid += 1
        #         else:
        #             smiles.append(Chem.CanonSmiles(smile))
        #     except:
        #         invalid += 1
        #         smiles.append(temp_str.strip()+"_invalid1")
    print("Invalid SMILES: {} / {}, Validity: {:2.2%}".format(invalid, len(hot_x), 1-invalid/len(hot_x)))
    return smiles

def thermal_argmax(prob_arr, temperature):
    prob_arr = np.log(prob_arr) / temperature
    prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
    print(prob_arr)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn("Probabilities to sample add to more than 1, {}".
                     format(prob_arr.sum()))
        prob_arr = prob_arr / (prob_arr.sum() + .0000000001)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn("Probabilities to sample still add to more than 1")
    return np.argmax(np.random.multinomial(1, prob_arr, 1))


def load_smiles(smi_file, max_len=None, return_filtered=False):
    if smi_file[-4:] == ".pkl":
        with open(smi_file, "rb") as f:
            smiles = pkl.load(f)
    else:  # assume file is a text file
        with open(smi_file, "r") as f:
            smiles = f.readlines()
        smiles = [i.strip() for i in smiles]

    if max_len is not None:
        if return_filtered:
            smiles, filtrate = filter_valid_smiles_return_invalid(
                smiles, max_len)
            if len(filtrate) > 0:
                print("Filtered {} smiles due to length".format(len(filtrate)))
            return smiles, filtrate

        else:
            old_len = len(smiles)
            smiles = filter_valid_length(smiles, max_len)
            diff_len = old_len - len(smiles)
            if diff_len != 0:
                print("Filtered {} smiles due to length".format(diff_len))

    return smiles


def load_string_and_data_df(data_file, max_len, string_type, data_path, reg_tasks=None, dtype="float64"):
    df = pd.read_csv(data_file, usecols=["index", "name", "canon_smiles"] + reg_tasks)
    df["smiles"] = df.canon_smiles.str.strip()
    df = df.drop(columns=["canon_smiles"])
    if string_type == "smiles":
        df["string_len"] = df.smiles.apply(lambda x: len(list(x)))
    # elif string_type == "selfies":
    #     df["selfies"] = df.smiles.apply(lambda x: selfies.encoder(x))
    #     df["string_len"] = df.selfies.apply(lambda x: selfies.len_selfies(x))
    # elif string_type == "modify_deepsmiles":
    #     converter = deepsmiles.Converter(rings=True, branches=True)
    #     df["deepsmiles"] = df.smiles.apply(lambda x: converter.encode(x))
    #     df["modify_deepsmiles"] = df.deepsmiles.apply(lambda x: modify_deepsmiles(x))
    #     df["string_len"] = df.modify_deepsmiles.apply(lambda x: len(x))
    # elif string_type == "deepsmiles":
    #     converter = deepsmiles.Converter(rings=True, branches=True)
    #     df["deepsmiles"] = df.smiles.apply(lambda x: converter.encode(x))
    #     df["string_len"] = df.deepsmiles.apply(lambda x: len(list(x)))
    elif string_type == "modify_smiles":
        df["modify_smiles"] = df.smiles.apply(lambda x: modify_smiles(x))
        df["string_len"] = df.modify_smiles.apply(lambda x: len(x))
    df = df[df.string_len <= max_len]
    df.to_csv(data_path)
    index = df["index"].tolist()
    names = df["name"].tolist()
    smiles = df["smiles"].tolist()
    strings = df[string_type].tolist()
    targets = np.vstack(df[reg_tasks].values).astype("float64")
    return index, names, smiles, strings, targets

def load_string_DA(index, or_smiles, x, y, max_len, string_type, data_path):
    df = pd.DataFrame({"index": index, "or_smiles": or_smiles, "smiles": x, "target": y})
    # print(df)
    if string_type == "smiles":
        df["string_len"] = df.smiles.apply(lambda x: len(list(x)))
    # elif string_type == "selfies":
    #     df["selfies"] = df.smiles.apply(lambda x: selfies.encoder(x))
    #     df["string_len"] = df.selfies.apply(lambda x: selfies.len_selfies(x))
    # elif string_type == "modify_deepsmiles":
    #     converter = deepsmiles.Converter(rings=True, branches=True)
    #     df["deepsmiles"] = df.smiles.apply(lambda x: converter.encode(x))
    #     df["modify_deepsmiles"] = df.deepsmiles.apply(lambda x: modify_deepsmiles(x))
    #     df["string_len"] = df.modify_deepsmiles.apply(lambda x: len(x))
    # elif string_type == "deepsmiles":
    #     converter = deepsmiles.Converter(rings=True, branches=True)
    #     df["deepsmiles"] = df.smiles.apply(lambda x: converter.encode(x))
    #     df["string_len"] = df.deepsmiles.apply(lambda x: len(list(x)))
    elif string_type == "modify_smiles":
        df["modify_smiles"] = df.smiles.apply(lambda x: modify_smiles(x))
        df["string_len"] = df.modify_smiles.apply(lambda x: len(x))
    df = df[df.string_len <= max_len]
    df.to_csv(data_path)
    strings = df[string_type].tolist()
    smiles = df["smiles"].tolist()
    or_smiles = df["or_smiles"].tolist()
    new_index = df["index"].tolist()
    targets = np.array(df["target"].tolist())
    return new_index, or_smiles, smiles, strings, targets

def smiles2one_hot_chars(smi_list, max_len):
    # get all the characters
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(" ")

    return chars


def make_charset(smi_file, char_file):
    with open(smi_file, "r") as afile:
        unique_chars = set(afile.read())
    bad = ["\n", "\""]
    unique_chars = [c for c in unique_chars if c not in bad]
    unique_chars.append(" ")
    print("found {} unique chars".format(len(unique_chars)))
    astr = str(unique_chars).replace("\"", "\"")
    print(astr)

    with open(char_file, "w") as afile:
        afile.write(astr)

    return


# =================
# data parsing io functions
# ==================

def CheckSmiFeasible(smi):
    # See if you can make a smiles with mol object
    #    if you can't, then skip
    try:
        get_molecule_smi(Chem.MolFromSmiles(smi))
    except:
        return False
    return True


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
    return s.count("1") % 2 == 0 and s.count("2") % 2 == 0


def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)


def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)


def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

def charge_N(smi):
    num_Nplus = len(smi) - len(smi.replace("[N+]", ""))
    num_equal = len(smi) - len(smi.replace("=", ""))
    charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
    return (charge == 0) & (num_Nplus <= 4 * num_equal)
