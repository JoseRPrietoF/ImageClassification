import random, shutil, os, glob
try:
    from page import PAGE
except:
    from data.page import PAGE
import re
import numpy as np

imgs_path = "/data/SimancasSearch/all"
page_path = "/data/SimancasSearch/all_regions/"
parts = ["test", "validation", "train"]
path_save = "SiSe/all_text_DAN"
path_files = "SiSe/all/"

def get_files(path_files:str):
    file_list = {}
    for c in os.listdir(path_files):
        print(c)
        p = os.path.join(path_files, c)
        aux = []
        fs = glob.glob(os.path.join(p, "*"))
        for f in fs:
            f = "_".join(f.split(".")[0].split("/")[-1].split("_")[:-1])
            aux.append(f)
        file_list[c] = aux
    return file_list

def normalize_text(text):
    sep = '$'
    text = text.lower()
    text = " ".join([x.split(sep)[0] for x in text.split(" ")])
    # text = re.sub(r'[^\w\s]','', text)
    text = re.sub(r'\d+','',text)
    text = re.sub(' +', ' ', text)
    # text = text.strip()
    return text

def main():
    # ABERTA 0 0.00678059 1007 1711 90 57 TextLine_1_1  6 0.0067778 7 2.79036e-06
    os.makedirs(path_save, exist_ok=True)
    for part in parts:
        print(f"========== {part} ==========")
        path_save_part = os.path.join(path_save, part)
        os.makedirs(path_save_part, exist_ok=True)
        path_get = os.path.join(path_files, part)
        file_list = get_files(path_get)
        for c, ls in file_list.items():
            # path_get_c = os.path.join(path_save_part, c)
            # os.makedirs(path_get_c, exist_ok=True)
            for num in ls:
                l = "_".join(num.split("_")[:-1])
                xml = os.path.join(page_path, f"{l}.xml")
                page = PAGE(xml)
                acts = page.get_textRegionsActs(GT=True)
                f = open(os.path.join(path_save_part, f"{num}_{c}.txt"), "w")
                for act in acts:
                    coords, name, info = act
                    text = info['text']
                    text = normalize_text(text)
                    
                    if name == num:
                        # if "CCA_CED_0233_0187" in name:
                        #     print(name)
                        f.write(f"{text}")
                f.close()
        
if __name__ == "__main__":
    main()