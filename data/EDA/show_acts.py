import cv2
import numpy as np, json
import tqdm
import pickle as pkl
import glob, os, copy, re, json
import os
import sys
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
from page import PAGE
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import argparse

colors = {
    'AM': (0,255,0),
    'M': (0,255,0),
    'I': (255,0,0),
    'AI': (255,0,0),
    'F': (0,0,255),
    'AF': (0,0,255),
    'C': (200,0,200),
    'AC': (200,0,200),
}

def read_list_sise(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    order = {}
    for line in lines:
        id_act, o = line.split()[:2]
        order[id_act] = o
    return order

def main(args):
    if not os.path.exists(args.path_save):
        os.mkdir(args.path_save)
    xmls = glob.glob(os.path.join(args.folder_xmls, "*xml"))
    class_PD = False
    if args.path_results != "no":
        class_PD = read_list_sise(args.path_results)
    for xml in xmls:
        page = PAGE(xml)
        fname = xml.split("/")[-1].split(".")[0]
        acts = page.get_textRegionsActs(max_iou=args.max_iou_acts, GT=args.GT)

        img = cv2.imread(os.path.join(args.folder_imgs, f"{fname}.jpg"))
        back = img.copy()
        for act_i, (coords, id_act, info) in enumerate(acts):
            print(id_act)
            
            # color_act = colors[char]
            type_ = info["type"]
            if class_PD:
                type_ = class_PD[id_act]
            color_act = colors.get(type_)
            cv2.drawContours(back, [coords], 0, color_act, -1)

            # blend with original image
            alpha = 0.25
            img = cv2.addWeighted(img, 1-alpha, back, alpha, 0)
            # cv2.drawContours(input_img_line, [contours], -1, (0, 0, 255,50), -1)
        path_save_img = os.path.join(args.path_save, f'{fname}.jpg')
        cv2.imwrite(path_save_img, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--GT', type=str, help='folder')
    parser.add_argument('--folder_xmls', type=str, help='folder')
    parser.add_argument('--folder_imgs', type=str, help='folder')
    parser.add_argument('--path_results', type=str, help='folder', default="")
    parser.add_argument('--path_save', type=str, help='algorithm')
    parser.add_argument('--max_iou_acts', type=float, help='algorithm', default=0.7)
    
    args = parser.parse_args()
    args.GT = args.GT.lower() in ["si", "yes", "y", "true"]
    main(args)