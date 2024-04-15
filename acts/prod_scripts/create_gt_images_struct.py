import os, random, shutil

path_imgs = "/data/carabela_segmentacion/FG_1574_images/"
path_save_all = "/data/carabela_segmentacion/FG_1574_gt"
path_gt = "FG_1574_gt"
perc_val = 0.15
follow_symlinks = True

def read_files(path:str):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    res_dict, res_ord_list = {}, []
    last = "F"
    for line in lines:
        fname, c = line.strip().split(" ")
        res_ord_list.append([fname, c])
        res_dict[fname] = c
        if c == "I" and last != "F":
            raise Exception(f"Inconsistencia con página {fname}")
        last = c
    return res_dict, res_ord_list


if __name__ == "__main__":
    total_images, total_images_tr, total_images_te = 0, 0, 0
    res_dict, res_ord_list = read_files(path_gt)
    path_dest = f"{path_save_all}/test"
    path_dest_tr = f"{path_save_all}/train"
    path_dest_val = f"{path_save_all}/validation"
    [os.makedirs(os.path.join(path_dest, c), exist_ok=True) for c in ["I", "M", "F"]]
    [os.makedirs(os.path.join(path_dest_tr, c), exist_ok=True) for c in ["I", "M", "F"]]
    [os.makedirs(os.path.join(path_dest_val, c), exist_ok=True) for c in ["I", "M", "F"]]
    i_count = 0
    for i, (fname, c) in enumerate(res_ord_list):
        if c == "I":
            i_count += 1
        path_img = os.path.join(path_imgs, f"{fname}.jpg")
        if random.random() >= perc_val:
            shutil.copyfile(path_img, os.path.join(path_dest_tr, c, f"{fname}.jpg"), follow_symlinks=follow_symlinks)
            total_images += 1
            total_images_tr += 1
        else:
            shutil.copyfile(path_img, os.path.join(path_dest, c, f"{fname}.jpg"), follow_symlinks=follow_symlinks)
            shutil.copyfile(path_img, os.path.join(path_dest_val, c, f"{fname}.jpg"), follow_symlinks=follow_symlinks)
            total_images += 1
            total_images_te += 1
    print(f"{i_count} deeds with {total_images} total images [{total_images_tr} tr {total_images_te} te]")