import os
import glob
try:
    from page import PAGE
except:
    from data.page import PAGE

# dest_file = "/home/jose/projects/image_classif/acts/results/chancery/train_gt"
dest_file = "/home/jose/projects/image_classif/acts/results/chancery/test_gt"
files = "/data/chancery2/labelled_volumes/test_page_4classes"
# files = "/data/chancery2/labelled_volumes/train_page_4classes"

def main():
    file_list = []
    p = os.path.join(files, "*.xml")
    fs = glob.glob(p)
    fs.sort()
    for file in fs:
        page = PAGE(file)
        acts = page.get_textRegionsActs(GT=True)
        fname = file.split("/")[-1]
        name_img = fname.split(".")[0]
        for act in acts:
            coords, name, info = act
            id_ = f"{name_img}_{name}"
            # print(info)
            # id_ = id_.replace("R_A","Z_A")
            file_list.append((id_, info["type"]))
            
    file_list.sort()
    f = open(dest_file, "w")
    for fname, c in file_list:
        # fname = fname.replace("Z_A", "R_A")
        f.write(f"{fname} {c}\n")
    f.close()

if __name__ == "__main__":
    main()