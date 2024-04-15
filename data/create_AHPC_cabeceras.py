import os, shutil, glob, random
import cv2

part_te = 0.2
part_dev = 0.1
path_data = "/data/carabela_segmentacion/pruebas_encabezado/crop-imgs"
dest = "AHPC_cabeceras"
classes="P,CP,O,A,T,V,R,CEN,DP,D,C,TH".split(",")
follow_symlinks = True
# to_RGB = True

def main():
    imgs = glob.glob(os.path.join(f"{path_data}", "*.png"))
    # classes = set([x.split(".")[0].split("_")[-1] for x in imgs])
    random.shuffle(imgs)
    imgs = [x for x in imgs if x.split(".")[0].split("_")[-1] in classes]
    n_test = int(len(imgs) * part_te)
    n_dev = int(len(imgs) * part_dev)
    te = imgs[:n_test]
    dev = imgs[n_test:n_test+n_dev]
    tr = imgs[n_test+n_dev:]
    print(len(imgs), len(tr), len(dev), len(te))
    print(classes)
    os.makedirs(dest, exist_ok=True)
    path_tr = os.path.join(dest, "train")
    path_te = os.path.join(dest, "test")
    path_dev = os.path.join(dest, "validation")
    os.makedirs(path_tr, exist_ok=True)
    os.makedirs(path_te, exist_ok=True)
    os.makedirs(path_dev, exist_ok=True)
    for c in classes:
        p = os.path.join(path_tr, c)
        os.makedirs(p, exist_ok=True)
        p = os.path.join(path_te, c)
        os.makedirs(p, exist_ok=True)
        p = os.path.join(path_dev, c)
        os.makedirs(p, exist_ok=True)
    for img in tr:
        img_path = img.split("/")[-1]
        c = img.split(".")[0].split("_")[-1]
        if c in classes:
            path_img = os.path.join(path_tr, c, img_path)
            img = cv2.imread(img)
            # print(img.shape)
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path_img, img) 
            # shutil.copyfile(img, path_img, follow_symlinks=follow_symlinks)
    for img in te:
        img_path = img.split("/")[-1]
        c = img.split(".")[0].split("_")[-1]
        if c in classes:
            path_img = os.path.join(path_te, c, img_path)
            img = cv2.imread(img)
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path_img, img) 
            # shutil.copyfile(img, path_img, follow_symlinks=follow_symlinks)
    for img in dev:
        img_path = img.split("/")[-1]
        c = img.split(".")[0].split("_")[-1]
        if c in classes:
            path_img = os.path.join(path_dev, c, img_path)
            img = cv2.imread(img)
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path_img, img) 
            # shutil.copyfile(img, path_img, follow_symlinks=follow_symlinks)


if __name__ == "__main__":
    main()