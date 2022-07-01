import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from rdkit import Chem


token_list = ["(Cl)", "(Br)", "(F)", "(I)", "(C)", "(O)", "(N)", "(=O)", "([O-])", "(=S)",
                     "(c1ccccc1)", "(c2ccccc2)", "(c3ccccc3)", "(c4ccccc4)",
                     "Cl", "Br", "[O-]", "[N-]",
                     "[C@H]", "[C@@H]",
                     "[NH3+]", "[N+]", "[nH]", "[n+]",
                     "[Si]", "[SiH]", "[SiH2]", "[SiH3]", "[PH]",
                     "c1ccccc1", "c2ccccc2", "c3ccccc3", "c4ccccc4", "c%99ccccc%99", "c%98ccccc%98", "c%97ccccc%97"]

ben_list = ["c1ccccc1", "c2ccccc2", "c3ccccc3", "c4ccccc4", "c%99ccccc%99", "c%98ccccc%98", "c%97ccccc%97"]
bran_ben_list = ["(c1ccccc1)", "(c2ccccc2)", "(c3ccccc3)", "(c4ccccc4)", "(c%99ccccc%99)", "(c%98ccccc%98)", "(c %97ccccc%97)"]



def modify_smiles(smile, token_dict=None, char_dict=None):

    or_smile = smile
    smile = [smile]
    for token in token_list:
        new_smile = []
        for smi in smile:
            if smi != "(" + token + ")":
                strings = smi.split(token)
                strings_len = len(strings)
                new_strings = []
                for (i, string) in enumerate(strings):
                    new_strings.append(string)
                    if i < strings_len-1:
                        new_strings.append(token)
                new_smile += new_strings
            else:
                new_smile += [smi]
        smile = new_smile
        while "" in smile:
            smile.remove("")
    # print(smile)
    if token_dict is not None:
        for i in smile:
            token_dict[i]

    new_smile = []
    for smi in smile:
        if smi in token_list:
            new_smile += [smi]
        else:
            new_smile += list(str(smi))

    new_smile = [i if i not in ben_list else "[C6H6]" for i in new_smile]
    new_smile = [i if i not in bran_ben_list else "[Bran_C6H6]" for i in new_smile]

    if char_dict is not None:
        for i in new_smile:
            char_dict[i]


    # print("length of orginal and new smiles string:", len(or_smile), "and", len(new_smile))
    if (token_dict is not None) and (char_dict is not None):
        return new_smile, token_dict, char_dict
    else:
        return new_smile


inverse_list = ["[C6H6]", "[Bran_C6H6]"]


def inverse_modify_smiles(modify_smile):
    modify_smile = [modify_smile]
    start = 99
    for token in inverse_list:
        new_modify_smile = []
        for modify_smi in modify_smile:
            strings = modify_smi.split(token)
            strings_len = len(strings)
            new_strings = []
            for (i, string) in enumerate(strings):
                new_strings.append(string)
                if i < strings_len-1:
                    ben = "c%" + str(start) + "ccccc%" + str(start)
                    if token == "[Bran_C6H6]":
                        ben = "(" + ben + ")"
                    new_strings.append(ben)
                    start -= 1
            new_modify_smile += new_strings
        modify_smile = new_modify_smile
        while "" in modify_smile:
            modify_smile.remove("")
    inverse_smiles = "".join(modify_smile)
    # inverse_smiles = Chem.CanonSmiles(inverse_smiles)
    return inverse_smiles

if __name__ == "__main__":

    """df = pd.read_csv("C:\zwang2020Doc\Tutorial\TASK1_organic_solvent/activity_coefficient_v1.csv")
    smiles = df.canon_smiles.tolist()"""

    smiles = "CCCC"
    smiles = Chem.CanonSmiles(smiles)
    print(smiles)
    print(len(modify_smiles(smiles)))
    print("".join(modify_smiles(smiles)))



    """token_dict = defaultdict(lambda: len(token_dict))
    char_dict = defaultdict(lambda: len(char_dict))

    new_smiles = []
    for smile in smiles:
        new_smile, token_dict, char_dict = modify_smiles(smile, token_dict, char_dict)
        new_smiles += new_smile
    char_dict = dict(char_dict)
    print(len(char_dict), char_dict)
    print(Counter(new_smiles))"""

    inverse_modify_smiles("c1ccc(C[C6H6])cc1")

