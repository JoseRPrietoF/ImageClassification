import os, random
import shutil

path_imgs = "/home/jose/projects/image_classif/data/JMBD4949_4950/imgs_"
path_gt = "/home/jose/projects/LLMs/datasets_own/AHPC/res_classif_"
folder = "4952"
if folder == "4949":
    path_imgs = "/data/carabela_segmentacion/idxs_JMBD4949/JMBD_"
elif folder == "4950":
    path_imgs = "/data/carabela_segmentacion/idxs_JMBD4950/JMBD_"

path_save_all = "/data/AHPC_book_segm/1folder/curve"
# path_save_all = "/home/jose/projects/image_classif/data/JMBD/1folder/curve"
perc_val = 0.15
follow_symlinks = True

num_groups = [8, 16, 32, 64, 128, 256, 512]

def load_groups(path:str):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        #50 65 CEN
        pini, pfin, c = line.strip().split(" ")
        res.append((int(pini), int(pfin), c))
    return res

def main():
    dict_p = load_groups(path=path_gt + folder)
    print(len(dict_p))
    for num_g in num_groups:
        path_dest = f"{path_save_all}/JMBD{folder}_{num_g}/test"
        path_dest_tr = f"{path_save_all}/JMBD{folder}_{num_g}/train"
        path_dest_val = f"{path_save_all}/JMBD{folder}_{num_g}/validation"
        [os.makedirs(os.path.join(path_dest, c), exist_ok=True) for c in ["I", "M", "F"]]
        [os.makedirs(os.path.join(path_dest_tr, c), exist_ok=True) for c in ["I", "M", "F"]]
        [os.makedirs(os.path.join(path_dest_val, c), exist_ok=True) for c in ["I", "M", "F"]]
        total_images = 0
        total_images_tr, total_images_te = 0, 0
        for pini, pfin, c in dict_p[:num_g]:
            i = pini
            path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
            c = "I"
            if not os.path.exists(path_page):
                raise Exception(f"PAge {path_page} does not exist for class {c}")
            if random.random() >= perc_val:
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                total_images += 1
                total_images_tr += 1
            else:
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                total_images += 1
                total_images_te += 1
            for i in range(pfin, pini, -1):
                i = pfin
                c = "F"
                path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
                if os.path.exists(path_page):
                    # raise Exception(f"PAge {path_page} does not exist for class {c}")
                    if random.random() >= perc_val:
                        shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                        total_images += 1
                        total_images_tr += 1
                    else:
                        shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                        shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                        total_images += 1
                        total_images_te += 1
                    break
            c = "M"
            for i in range(pini+1, pfin):
                path_page = os.path.join(path_imgs + folder, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")
                if not os.path.exists(path_page):
                    continue
                if random.random() >= perc_val:
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_tr, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                    total_images += 1
                    total_images_tr += 1
                else:
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                    shutil.copyfile(path_page, os.path.join(path_page, os.path.join(path_dest_val, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")), follow_symlinks=follow_symlinks)
                    total_images += 1
                    total_images_te += 1
        print(f"Group with {num_g} deeds with {total_images} total images [{total_images_tr} tr {total_images_te} te]")

if __name__ == "__main__":
    main()