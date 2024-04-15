import glob, os

path_save = "/data/chancery2/labelled_volumes/idxs/all"
path_segments = ["/data/chancery2/labelled_volumes/idxs/32segments/test", "/data/chancery2/labelled_volumes/idxs/32segments/train"]

def read_txts(l:list):
    res = []
    for fname in l:
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            res.append(line.strip())
    return res
    
def main():
    
    dict_names = {}
    for path_segm in path_segments:
        files = glob.glob(os.path.join(path_segm, "*txt"))
        for f in files:
            fname = f.split("-")[0].split("/")[-1]
            name = dict_names.get(fname, [])
            name.append(f)
            dict_names[fname] = name
    for name_folder, files in dict_names.items():
        fname_idx = os.path.join(path_save, f"{name_folder}.idx")
        f = open(fname_idx, "w")
        idxs = read_txts(files)
        for idx in idxs:
            f.write(f"{idx}\n")
        f.close()

if __name__ == "__main__":
    main()