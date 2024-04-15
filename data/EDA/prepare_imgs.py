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

def main(args):
    if not os.path.exists(args.path_save):
        os.mkdir(args.path_save)
    imgs_gt = glob.glob(os.path.join(args.path_GT, "*jpg"))
   
    for img_gt in imgs_gt:
        fname = img_gt.split("/")[-1]
        img1_path = os.path.join(args.path_results_1, fname)
        img2_path = os.path.join(args.path_results_2, fname)

        img_gt = cv2.imread(img_gt)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        # image = cv2.resize(image, (0, 0), None, .25, .25)
        horizontal = np.hstack((img_gt, img1, img2))

        path_save_img = os.path.join(args.path_save, f'{fname}')
        cv2.imwrite(path_save_img, horizontal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--path_results_1', type=str, help='folder')
    parser.add_argument('--path_results_2', type=str, help='folder')
    parser.add_argument('--path_GT', type=str, help='folder')
    parser.add_argument('--path_save', type=str, help='folder', default="")
    
    args = parser.parse_args()
    main(args)