import os, glob, shutil
import cv2

min_rw = 30

def running_idx(files_idx:str):
    acc_prob = 0
    with open(files_idx, "r") as f:
        lines = f.readlines()
        for line in lines:
            prob = float(line.split(" ")[2])
            acc_prob += prob
    return acc_prob


def main(dir_path:str, start_page:int, save_dir:str):
    files_idx = glob.glob(os.path.join(dir_path, "*.idx"))
    files_idx.sort()
    for file_idx in files_idx[start_page:]:
        running_file = running_idx(file_idx)
        file_name = file_idx.split(".")[0]
        # if running_file > min_rw:
        #     print(file_name, running_file)
        # else:
        #     print(file_name, running_file, "    ********** se borra **********")
        if running_file > min_rw:
            file_jpg = file_name + ".jpg"
            fname = file_name.split("/")[-1]
            file_jpg_dest = os.path.join(save_dir,  fname + ".jpg" )
            # os.symlink(file_jpg, file_jpg_dest)
            img = cv2.imread(file_jpg, 0) 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # shutil.copy(file_jpg, file_jpg_dest)
            cv2.imwrite(file_jpg_dest, img)


if __name__ == "__main__":
    print(f"min_rw {min_rw}")
    dirs = [
            ["/data/carabela_segmentacion/prod/JMBD_4946",49, "/home/jose/projects/image_classif/data/JMBD4949_4950/prod_4946/test/I"], 
            ["/data/carabela_segmentacion/prod/JMBD_4952",49, "/home/jose/projects/image_classif/data/JMBD4949_4950/prod_4952/test/I"]
    ]
    for dir, start_page, save_dir in dirs:
        main(dir, start_page, save_dir)