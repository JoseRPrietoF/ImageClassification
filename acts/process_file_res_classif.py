import os, numpy as np

def load_res(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    first = [x.upper() for x in lines[0].strip().split(" ")[3:]]
    lines = lines[1:]
    res = []
    for line in lines:
        line = line.strip()
        line_prob = [float(x) for x in line.split(" ")[2:]]
        fname = line.split(" ")[0]
        ini, fin = fname.split("pages_")[-1].split("_")[0].split("-")
        ini, fin = int(ini), int(fin)
        c = np.argmax(line_prob)
        prob = line_prob[c]
        c = first[c]
        res.append((ini, fin, c, prob))
    return res


def main(path:str, path_res_name:str):
    res = load_res(path)
    res.sort()
    f = open(path_res_name, "w")
    for ini, fin, c, prob in res:
        f.write(f"{ini} {fin} {c} {prob}\n")
    f.close()

if __name__ == "__main__":
    names = ["4946", "4952"]
    path = "/home/jose/projects/docClasifIbPRIA22/works_prod_JMBD{}_groups_openset1vsall/work_128,128_numFeat2048/results_prod.txt"
    path_res = "."
    for name in names:
        path_res_name = os.path.join(path_res, f"res_classif_{name}")
        main(path.format(name), path_res_name)
