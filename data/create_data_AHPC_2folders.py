import random, shutil, os, glob

combinations = [
    ("4946", "4949"),
    ("4946", "4950"), 
    ("4946", "4952"),
    ("4949", "4950"),
    ("4949", "4952"),
    ("4950", "4952")
]
perc_val = 0.15
path_orig = f"/home/jose/projects/image_classif/data/JMBD/"
path_dest = f"/home/jose/projects/image_classif/data/JMBD/2folders"

classes = ["I", "M", "F"]
follow_symlinks = True

def create_link(p, p_dest):
    try:
        # os.symlink(p, p_dest)
        shutil.copyfile(p, p_dest, follow_symlinks=follow_symlinks)
    except FileExistsError as e:
        pass

def main():
    # os.symlink(path_page, os.path.join(path_page, os.path.join(path_dest, c, f"JMBD_{folder}_{str(i).zfill(5)}.jpg")))
    for folder in combinations:
        folder_name = f"JMBD_tr_{'_'.join(folder)}"
        fold_base = os.path.join(path_dest, folder_name)
        fold_tr, fold_te, fold_val = os.path.join(fold_base, "train"), os.path.join(fold_base, "test"), os.path.join(fold_base, "validation")
        for c in classes:
            tr_path = os.path.join(fold_tr, c)
            val_path = os.path.join(fold_val, c)
            te_path = os.path.join(fold_te, c)
            os.makedirs(tr_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            os.makedirs(te_path, exist_ok=True)
            for fold in folder:
                path_orig_fold_c = os.path.join(path_orig, f"JMBD{fold}", "train", c,  "*jpg")
                files = glob.glob(path_orig_fold_c)
                random.shuffle(files)
                num_val = int(len(files) * 0.2)
                for p in files[:num_val]:
                    fname = p.split("/")[-1]
                    p_dest = os.path.join(val_path, fname)
                    create_link(p, p_dest)
                    p_dest = os.path.join(te_path, fname)
                    create_link(p, p_dest)
                for p in files[num_val:]:
                    fname = p.split("/")[-1]
                    p_dest = os.path.join(tr_path, fname)
                    create_link(p, p_dest)

if __name__ == "__main__":
    main()