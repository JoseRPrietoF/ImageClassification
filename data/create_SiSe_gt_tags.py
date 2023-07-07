import os
import glob

dest_file = "/home/jose/projects/image_classif/acts/results/SiSe/test_all_gt"

# files = "/home/jose/projects/image_classif/data/SiSe/cut/test"
files = "/home/jose/projects/image_classif/data/SiSe/all/test"

def main():
    file_list = []
    for c in os.listdir(files):
        print(c)
        p = os.path.join(files, c)
        fs = glob.glob(os.path.join(p, "*"))
        for file in fs:
            fname = file.split("/")[-1]
            file_list.append((fname, c))
    file_list.sort()
    f = open(dest_file, "w")
    for fname, c in file_list:
        f.write(f"{fname} {c}\n")
    f.close()

if __name__ == "__main__":
    main()