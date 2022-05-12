import cv2
import glob, os
from sklearn.utils import shuffle
import tqdm

def read_csv(fpath, ktype):
    f = open(fpath, "r", encoding='utf-8', errors='ignore')
    if ktype == "49":
        lines = f.readlines()[:-1]
    else:
        lines = f.readlines()[1:]
    f.close()
    res = []
    for i, line in enumerate(lines):
        # print("line ", i+1)
        if ktype == "49":
            line = line.split(",")
            p_ini = line[1]
            p_fin = line[2]
            tip_abreviada = line[4].lower()
        else:
            line = line.split(",")
            p_ini = line[0]
            p_fin = line[1]
            tip_abreviada = line[3].lower()
        p_ini = int(p_ini)
        p_fin = int(p_fin)
        
        tip_abreviada = tip_abreviada.replace(" ", "")
        tip_abreviada = tip_abreviada.replace("?", "unk")
        # tip_abreviada = tip_abreviada.replace("s?", "s")
        # if tip_abreviada == "?":
        #     continue
        if "-" in tip_abreviada:
            tip_abreviada = tip_abreviada.split("-")[0]
        if tip_abreviada == "s":
            tip_abreviada = "p"
        res.append((tip_abreviada, p_ini, p_fin))
    return res

def init_JMBD4949(csv_path, img_dirs):
        ktype = "49"
        files = read_csv(csv_path, ktype=ktype)
        # if ktype == "49":
        #     file_names = glob.glob(os.path.join(self.img_dirs, "*.tif"))
        # else:
        #     file_names = glob.glob(os.path.join(self.img_dirs, "*.jpg"))
        res = []
        for tip_abreviada, p_ini, p_fin in files:
            # print(tip_abreviada, p_ini, p_fin)
            for num in range(p_ini, p_fin+1):
                fpath_img = os.path.join(img_dirs, f"JMBD_4949_{num:05}.jpg")
                if num == p_ini:
                    c = "I"
                elif num == p_fin:
                    c = "F"
                else:
                    c = "M"
                res.append((fpath_img, c))
        return res

def create_structure(path_res, classes):
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    tr, te, val = os.path.join(path_res, "train"), os.path.join(path_res, "test"), os.path.join(path_res, "validation")
    if not os.path.exists(tr):
        os.mkdir(tr)
    if not os.path.exists(te):
        os.mkdir(te)
    if not os.path.exists(val):
        os.mkdir(val)
    for c in classes:
        for t in [tr,te,val]:
            p = os.path.join(t, c)
            if not os.path.exists(p):
                os.mkdir(p)

def load(fpath, height, width):
    img = cv2.imread(fpath)
    if height is not None and width is not None:
        img = cv2.resize(img, (width, height)) 
    return img

if __name__ == "__main__":
    corpus = "JMBD4949"
    classes = ["I", "M", "F"]
    img_dirs = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_4949"
    csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4949_Clasificacion_20220405_2.csv"
    # csv_path = "/data/carabela_segmentacion/csv_gt/JMBD_4950_Clasificacion_20220405.csv"
    path_res = "/home/jose/projects/image_classif/data/JMBD4949"
    auto_split = [0.60,0.10,0.30] # tr val test
    width, height = 768, 1024
    random_state=0
    create_structure(path_res, classes)
    
    if corpus == "JMBD4949":
        res = init_JMBD4949(csv_path=csv_path, img_dirs=img_dirs)
        if auto_split is not None:
            num_tr = int(len(res) * auto_split[0])
            num_val = int(len(res) * auto_split[1])
            num_test = len(res) - num_tr - num_val
            res = shuffle(res, random_state=random_state)
            tr_data = res[:num_tr]
            val_data = res[num_tr:num_tr+num_val]
            te_data = res[num_tr+num_val:]
            print(num_tr, num_val, num_test, len(tr_data), len(val_data), len(te_data))
            for fpath_img, c in tqdm.tqdm(tr_data):
                path_folder = os.path.join(path_res, "train", c)
                fname = fpath_img.split("/")[-1].split(".")[0]
                path_save = os.path.join(path_folder, fname+".jpg")
                img = load(fpath_img, width=width, height=height)
                cv2.imwrite(path_save, img)
            for fpath_img, c in tqdm.tqdm(te_data):
                path_folder = os.path.join(path_res, "test", c)
                fname = fpath_img.split("/")[-1].split(".")[0]
                path_save = os.path.join(path_folder, fname+".jpg")
                img = load(fpath_img, width=width, height=height)
                cv2.imwrite(path_save, img)
            for fpath_img, c in tqdm.tqdm(val_data):
                path_folder = os.path.join(path_res, "validation", c)
                fname = fpath_img.split("/")[-1].split(".")[0]
                path_save = os.path.join(path_folder, fname+".jpg")
                img = load(fpath_img, width=width, height=height)
                cv2.imwrite(path_save, img)
            