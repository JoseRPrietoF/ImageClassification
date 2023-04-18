import os, glob, re, tqdm
import numpy as np
import pickle as pkl

def read_file_imf(fpath:str):
    f = open(fpath, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        if line[1] in ["I", "M", "F"]:
            res.append((line[0], line[1]))
    return res

def get_groups(p:str) -> list:
    res = []
    res_imfs = read_file_imf(p)
    group = []
    pini, pfin = "", ""
    for fname, c in res_imfs:
        if c == "I":
            group = []
            pini = int(fname.split("_")[-1])
            group.append(fname)
        elif c == "M":
            group.append(fname)
        else:
            group.append(fname)
            pfin = int(fname.split("_")[-1])
            res.append((pini, pfin, group))
    return res

def read_tfidf_file(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines[1:]:
        word, *tfidf_v = line.strip().split(" ")
        tfidf_v = [float(x) for x in tfidf_v[:-1]]
        res[word] = tfidf_v
    return res

def create_group(group:list, path_idx:str):
    res = {}
    for p in group:
        f = open(os.path.join(path_idx, f"{p}.idx"), "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.split(" ")
            word = line[0]
            prob = float(line[2])
            res[word] = res.get(word, 0.0) + prob
    return res


def main(path_grupos:str, path_save:str, folder_name:str, path_idx:str):
    c = "unk"
    res_groups = get_groups(path_grupos)
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    res_groups = tqdm.tqdm(res_groups)
    for i, (ini, fin, group) in enumerate(res_groups):
        v = create_group( group, path_idx)
        path_save_f = os.path.join(path_save, f'{folder_name}_pages_{ini}-{fin}_{c}.idx')
        f = open(path_save_f, "w")
        for k,prob in v.items():
            f.write(f"{k} {prob}\n")
        f.close()
        


if __name__ == "__main__":
    name = "4946"
    folder_name = f"JMBD{name}"
    path_groups = f"/home/jose/projects/image_classif/acts/JMBD_{name}_gt"
    path_idx = f"/data/carabela_segmentacion/prod/JMBD_{name}"
    path_save = f"groups_gt/{folder_name}"
    main(path_groups, path_save, folder_name, path_idx)