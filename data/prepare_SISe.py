import random, shutil, os, glob
try:
    from page import PAGE
except:
    from data.page import PAGE
import cv2
import numpy as np

imgs_path = "/data/SimancasSearch/all"
classes = ["I", "M", "F", "C"]
paths = {
    "train": 
        {
             "page_dir": "/data/SimancasSearch/partitions/tr_regions/"
        },
    "test": 
        {
             "page_dir": "/data/SimancasSearch/partitions/te_regions/"
        }
    
    }

only_completes = False

path_save = "SiSe/all"

eval = 0.15

def get_files(path_files:str):
    all_files_page = glob.glob(os.path.join(path_files, '*xml'))    
    all_files_page.sort()
    max_len = 1
    res = {}
    for fpath in all_files_page:
        *name, num = fpath.split("_")
        num = int(num.split(".")[0])
        name = "_".join(name)
        for i in range(0, 1000):
            name2 = f'{name}_{i}'
            pages = res.get(name2, [])
            if len(pages) == 0 or pages[-1][0]+1 == num:
                pages.append((num, fpath))
                break
        res[name2] = pages
    for _, l in res.items():
        max_len = max(max_len, len(l))
    return res

def get_completes(group_list:list):
    aux = []
    num = 0
    opened, opened_at = False, 0
    for num_, xml in group_list:
        page = PAGE(xml)
        acts = page.get_textRegionsActs(GT=True)
        acts_aux = []
        for act in acts:
            coords, name, info = act
            type_ = info['type']
            if not opened and type_ in ["AF", "AM"]:
                continue
            elif type_ in ["AI"]:
                opened = True
                # opened_at = num+1
            elif type_ in ["AF"]:
                opened = False
            # AC -> added
            
            acts_aux.append(act)
            num += 1
        aux.append((num_, xml, acts_aux))
    if opened:
        for num_, xml, acts_aux in aux[::-1]:
            while acts_aux and acts_aux[-1][2]['type'] in ["AM", "AI"]:
                # print("si", acts_aux[-1])
                acts_aux.pop(len(acts_aux)-1)
            if acts_aux and acts_aux[-1][2]['type'] in ["AF", "AC"]:
                # print("no")
                break
        # print("opened")
    return aux

def get_all(group_list:list):
    aux = []
    num = 0
    for num_, xml in group_list:
        page = PAGE(xml)
        acts = page.get_textRegionsActs(GT=True)
        acts_aux = []
        for act in acts:
            acts_aux.append(act)
        aux.append((num_, xml, acts_aux))
    return aux

def main():
    os.makedirs(path_save, exist_ok=True)
    part = "validation"
    path_save_part = os.path.join(path_save, part)
    os.makedirs(path_save_part, exist_ok=True)
    dirs = {
        c:os.path.join(path_save_part, c) for c in classes
        }
    for v in dirs.values():
        os.makedirs(v, exist_ok=True)
    for part, info in paths.items():
        print(f"========== {part} ==========")
        path_save_part = os.path.join(path_save, part)
        path_save_part_eval = os.path.join(path_save, "validation")
        os.makedirs(path_save_part, exist_ok=True)
        dirs = {
            c:os.path.join(path_save_part, c) for c in classes
            }
        for v in dirs.values():
            os.makedirs(v, exist_ok=True)
        page_dir = info["page_dir"]
        files = get_files(page_dir)
        for group_name, group_list in list(files.items()):
            if only_completes:
                group_list = get_completes(group_list)
            else:
                group_list = get_all(group_list)
            for num_, xml, acts in group_list:
                # page = PAGE(xml)
                # acts = page.get_textRegionsActs(GT=True)
                
                fname = xml.split("/")[-1].split(".")[0]
                img_path = os.path.join(imgs_path, f"{fname}.jpg")
                # print(acts)
                # print(img_path)
                img = cv2.imread(img_path)
                #img[y:y+h, x:x+w]
                for act in acts:
                    coords, name, info = act
                    type_ = info['type'][-1]
                    # print(type_, name)
                    path_save_act = os.path.join(path_save_part, type_, name+f"_{type_}.jpg")
                    if part == "train":
                        if random.random() < eval:
                            path_save_act = os.path.join(path_save_part_eval, type_, name+f"_{type_}.jpg")
                            
                         

                    y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
                    x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
                    img_act = img[y_min:y_max, x_min:x_max]

                    cv2.imwrite(path_save_act, img_act)

if __name__ == "__main__":
    main()