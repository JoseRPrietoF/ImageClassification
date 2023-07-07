import os, random
import shutil

path_imgs = "/home/jose/projects/image_classif/data/JMBD4949_4950/imgs_"
path_gt = "/home/jose/projects/LLMs/datasets_own/AHPC/res_classif_"
folder = "4952"
if folder == "4949":
    path_imgs = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_"
elif folder == "4950":
    path_imgs = "/data/carabela_segmentacion/idxs_JMBD4950/JMBD_"
path_dest = f"/home/jose/projects/image_classif/data/JMBD/1folder/JMBD{folder}/test"
path_dest_tr = f"/home/jose/projects/image_classif/data/JMBD/1folder/JMBD{folder}/train"
path_dest_val = f"/home/jose/projects/image_classif/data/JMBD/1folder/JMBD{folder}/validation"

perc_val = 0.15
follow_symlinks = True

def load_groups(path:str):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines:
        #50 65 CEN
        pini, pfin, c = line.strip().split(" ")
        res[(int(pini), int(pfin))] = c
    return res

def main():
    dict_p = load_groups(path=path_gt + folder)
    print(dict_p)
    [os.makedirs(os.path.join(path_dest, c), exist_ok=True) for c in ["I", "M", "F"]]
    [os.makedirs(os.path.join(path_dest_tr, c), exist_ok=True) for c in ["I", "M", "F"]]
    [os.makedirs(os.path.join(path_dest_val, c), exist_ok=True) for c in ["I", "M", "F"]]
    for (pini, pfin), c in dict_p.items():
        i = pini
        path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
        c = "I"
        # print(path_page, c)
        if not os.path.exists(path_page):
            raise Exception(f"PAge {path_page} does not exist for class {c}")
        if random.random() >= perc_val:
            # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
            shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
        else:
            # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
            # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
            shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
            shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)

        for i in range(pfin, pini, -1):
            i = pfin
            c = "F"
            path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
            if os.path.exists(path_page):
                # raise Exception(f"PAge {path_page} does not exist for class {c}")
                if random.random() >= perc_val:
                    # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                else:
                    # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                    # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                break
            # print(path_page, c)


        c = "M"
        for i in range(pini+1, pfin):
            path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
            # print(path_page, c)
            if not os.path.exists(path_page):
                continue
                # raise Exception(f"PAge {path_page} does not exist for class {c}")
            if random.random() >= perc_val:
                # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
            else:
                # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
       


if __name__ == "__main__":
    main()