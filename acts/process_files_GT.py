import os, numpy as np, glob

def load_res(p:str):
    files = glob.glob(os.path.join(p, "*idx"))
    res = []
    for file in files:
        _, pages, c = file.split("/")[-1].split("_")
        ini, fin = pages.split("-")
        ini, fin = int(ini), int(fin)
        c = c.split(".")[0].upper()
        res.append((ini, fin, c))
    return res


def main(path:str, path_res_name:str, classif:bool, name:str):
    res = load_res(path)
    res.sort()
    f = open(path_res_name, "w")
    for ini, fin, c in res:
        if classif:
            f.write(f"{ini} {fin} {c} \n")
        else:
            f.write(f"JMBD_{name}_{ini:05} I \n")
            for i in range(ini+1, fin):
                f.write(f"JMBD_{name}_{i:05} M \n")
            f.write(f"JMBD_{name}_{fin:05} F \n")
    f.close()

if __name__ == "__main__":
    classif = False #false = segment
    names = ["4949", "4950"]
    path = "/data/carabela_segmentacion/idxs_JMBD{}/idxs_clasif_all_files"
    path_res = "."
    for name in names:
        if classif:
            path_res_name = os.path.join(path_res, f"res_classif_{name}")
        else:
            path_res_name = os.path.join(path_res, f"JMBD_{name}")
        main(path.format(name), path_res_name, classif, name)
