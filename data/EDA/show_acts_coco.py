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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import imagesize

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

dict_class = {
        "AI": 0, "I":0,
        "AM": 1, "M":1,
        "AF": 2, "F":2,
        "AC": 3, "C":3
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def evaluate(detections, ground_truths):
    with open('detections.json', 'w') as f:
        json.dump(detections, f, cls=NpEncoder)
    with open('ground_truths.json', 'w') as f:
        json.dump(ground_truths, f, cls=NpEncoder)

    coco_gt = COCO('ground_truths.json')
    coco_dt = coco_gt.loadRes('detections.json')
    # coco_gt = COCO('HisClimaDataset_test_coco_format.json')
    # coco_dt = coco_gt.loadRes('coco_instances_results.json')

    
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def create_coco_format(images, annotations, imfc = True, hyp=False):
    """
     "images": [
            {
                "id": 0,
                "license": 1,
                "file_name": "<filename0>.<ext>",
                "height": 480,
                "width": 640,
                "date_captured": null
            },
            ...
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 2,
                "bbox": [260, 177, 231, 199],
                "segmentation": [...],
                "area": 45969,
                "iscrowd": 0
            },
            ...
        ]
    """
    if imfc:
        categories =  [
            {"id": id_c, "name": name} for name, id_c in dict_class.items()
        ]
    else:
        categories =  [
            {
                "id": 0,
                "name": "text",
            },
        ]
    if not hyp:
        d = {
            "info": {
                "year": "2023",
                "version": "1.0",
                "description": "SiSe",
                "contributor": "Jose",
                "url": "",
                "date_created": "2023-01-19T09:48:27"
            },
            "licenses": [
                {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
                },
            ],
            "categories": categories,
            "images": images,
            "annotations": annotations
        }
    else:
        d = annotations
    return d

def read_list_sise(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    order = {}
    for line in lines:
        id_act, o = line.split()[:2]
        order[id_act] = o
    return order

def get_bbox(contours):
    xmax ,xmin = np.max([ x[0] for x in contours]), np.min([ x[0] for x in contours])
    ymax ,ymin = np.max([ x[1] for x in contours]), np.min([ x[1] for x in contours])
    return [xmin, ymin, xmax, ymax]

def get_segm(coords):
    """ 
    x = w, 
    y = h
    coords = [[x,y]]
    """
    xs = [x[0] for x in coords]
    ys = [x[1] for x in coords]
    return xs, ys

def main(args):
    xmls = glob.glob(os.path.join(args.GT, "*xml"))
    class_PD = read_list_sise(args.path_results)
    images = glob.glob(os.path.join(args.path_images, f"*jpg"))
    images_list = [{
                "id": i,
                "license": 1,
                "file_name": img_name,
                "height": imagesize.get(img_name)[1],
                "width": imagesize.get(img_name)[0],
            } for i, img_name in enumerate(images)]
    res = []
    for xml in xmls:
        page = PAGE(xml)
        fname = xml.split("/")[-1].split(".")[0]
        xml_path_hyp = os.path.join(args.folder_xmls, f"{fname}.xml")
        page_hyp = PAGE(xml_path_hyp)
        acts = page.get_textRegionsActs(max_iou=args.max_iou_acts, GT=True)
        acts_hyp = page_hyp.get_textRegionsActs(max_iou=args.max_iou_acts, GT=False)
        annotations_gt = []
        offset_acts = 0
        ## GT
        for act_i, (coords, id_act, info) in enumerate(acts):
            print(id_act)
            offset_acts += 1
            bbox = get_bbox(coords)
            # print(info["type"], bbox)
            coordsPr = Polygon(coords)
            pR = cascaded_union(coordsPr)
            coordsR = np.array(pR.exterior.coords)
            px, py = get_segm(coordsR)
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
            if args.imfc:
                cat_id = dict_class[info["type"]]
            else:
                cat_id = 1
            annotations_gt.append({
                "id": offset_acts,
                "image_id": act_i,
                "category_id":cat_id,
                "bbox": bbox,
                "segmentation": [poly],
                "area": coordsPr.area,
                "iscrowd": 0,
            })
        
        ##HYP
        annotations_noPD, annotations_PD = [], []
        offset_acts_hyp = 0
        for act_i, (coords, id_act, info) in enumerate(acts_hyp):
            print(id_act)
            offset_acts_hyp += 1
            bbox = get_bbox(coords)
            # print(info["type"], bbox)
            coordsPr = Polygon(coords)
            pR = cascaded_union(coordsPr)
            coordsR = np.array(pR.exterior.coords)
            px, py = get_segm(coordsR)
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x] # [(1,2), (3,4)] -> [1,2,3,4]
            # color_act = colors[char]
            type_noPD = info["type"]
            type_PD = class_PD[id_act]
            if args.imfc:
                cat_id_noPD = dict_class[type_noPD]
                cat_id_PD = dict_class[type_PD]
            else:
                cat_id_noPD = 1
                cat_id_PD = 1
            
            annotations_noPD.append({
                "id": offset_acts_hyp,
                "image_id": act_i,
                "category_id":cat_id_noPD,
                "bbox": bbox,
                "segmentation": [poly],
                "area": coordsPr.area,
                "iscrowd": 0,
                "score": 1
            })
            annotations_PD.append({
                "id": offset_acts_hyp,
                "image_id": act_i,
                "category_id":cat_id_PD,
                "bbox": bbox,
                "segmentation": [poly],
                "area": coordsPr.area,
                "iscrowd": 0,
                "score": 1
            })
        # print(f"\n\n  -------------------- {fname} --------------------  ")
        ground_truths = create_coco_format(images_list, annotations_gt, imfc=True)
        # print("Detection without PD")
        detections_noPD = create_coco_format(images_list, annotations_noPD, hyp=True, imfc=True)
        ev_noPD = evaluate(detections_noPD, ground_truths)
        # print("Detections with PD")
        detections_PD = create_coco_format(images_list, annotations_PD, hyp=True, imfc=True)
        ev_PD = evaluate(detections_PD, ground_truths)
        res.append((fname, ev_noPD[0], ev_PD[0]))
    res.sort()
    for fname, noPD, PD in res:
        print(f"{fname}  - noPD  {noPD:.2f}   PD {PD:.2f}")
    mean_noPD = np.mean([x[1] for x in res])
    mean_PD = np.mean([x[2] for x in res])
    print(f"Mean noPD : {mean_noPD:.4f}")
    print(f"Mean PD : {mean_PD:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--GT', type=str, help='folder')
    parser.add_argument('--folder_xmls', type=str, help='folder')
    parser.add_argument('--path_results', type=str, help='folder', default="")
    parser.add_argument('--max_iou_acts', type=float, help='algorithm', default=0.7)
    parser.add_argument('--path_images', type=str, help='algorithm')
    parser.add_argument('--imfc', type=str, help='algorithm', default="si")
    args = parser.parse_args()
    args.imfc = args.imfc.lower() in ["si", "y", "yes", "true"]
    main(args)