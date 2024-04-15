### obtaining the results
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
import argparse, re, os, sys, numpy as np
from PD import levenshtein
from get_results import sub_cost, read_IG
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
from data.page import PAGE
from acts.sequences_SiSe_all_pages import get_xmls

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_bbox(contours):
    xmax ,xmin = np.max([ x[0] for x in contours]), np.min([ x[0] for x in contours])
    ymax ,ymin = np.max([ x[1] for x in contours]), np.min([ x[1] for x in contours])
    return [xmin, ymin, xmax, ymax]

def append_text(vector:list, text, IG_order:dict):
    text = text.strip().replace("\n", " ")
    text = re.sub(' +', ' ', text).split(" ")
    for word in text:
        word = word.split("$")[0]
        if not word:
            continue
        prob = 1
        # word, prob = line[0].upper(), float(line[2])
        pos_word = IG_order.get(word, -1)
        # if pos_word is not None:
        vector[pos_word] += prob
    return vector

def get_regions(path, max_iou_acts = 0.7, get_text = False, is_GT=False):
    xmls = get_xmls(path)
    pages = {}
    res = {}
    # exit()
    for xml in xmls:
        name = xml.split("/")[-1].split(".")[0]
        page = PAGE(xml)
        pages[name] = page
        acts = page.get_textRegionsActs(max_iou=max_iou_acts, GT=is_GT)
        for coords, id_act, info in acts:
            if is_GT:
                id_act = f"{name}_{id_act}"
            res[id_act] = (coords, info)
    return res, pages

def create_acts_chancery(path_file:str,args=None, is_GT=False):
    f = open(path_file, "r")
    lines = f.readlines()
    f.close()
    
    res_groups = {}
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        fname = fname.split(".")[0]
        
        num_book = "_".join(fname.split("_")[:2])
        book = res_groups.get(num_book, [])
        book.append([fname, c])
        res_groups[num_book] = book
    
    acts = []
    for name_group, results_group in res_groups.items(): 
        aux = []
        for fname, c in results_group:
            aux.append(fname)
            if args.alg == "raw":
                if args.class_to_cut == c:
                    acts.append(aux)
                    aux = []
            else:
                if c in ["AF", "AC", "F", "C"]:
                    acts.append(aux)
                    aux = []
        if aux:
            acts.append(aux)
    return acts

def restore_groups(v_acts, acts):
    res = {}
    for group, v_group in zip(acts, v_acts):
        group_name = "_".join(group[0].split("_")[:2])
        list_g = res.get(group_name, [])
        list_g.append((group, v_group))
        res[group_name] = list_g
    return res


def append_prix(vector:list, prix:list, IG_order:dict):
    for word, prob in prix:
        pos_word = IG_order.get(word, -1)
        # if pos_word is not None:
        vector[pos_word] += prob
    return vector

def get_prix(page_path_prix:str, coords, IG_order):
    res = []
    polygon = Polygon(coords)
    with open(page_path_prix, "r") as f:
        lines = f.readlines()
        for line in lines:
            word, prob, x1, y1, x2, y2 = line.strip().split(" ")
            word = word.upper()
            if word in IG_order:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                point = Point(int( (x1+x2)/2 ), int( (y1+y2)/2 ))
                if polygon.contains(point):
                    res.append([word, float(prob)])
    return res

    
def create_vector_acts(regions_hyp:dict, acts:list, path_prix:str, IG_order:dict):
    res_acts = []
    for act in acts:
        vector = [0]*(len(IG_order)+1)
        for page in act:
            coords, info = regions_hyp.get(page)
            page_path_prix = os.path.join(path_prix, f"{'_'.join(page.split('_')[:-1])}.idx")
            prix = get_prix(page_path_prix, coords, IG_order)
            vector = append_prix(vector, prix, IG_order)
        vector = vector[:-1] # last component is the "residual" or other words outside of IG
        res_acts.append(vector)
    return res_acts
    
def main(args):
    regions_gt, pages_gt = get_regions(args.path_page_gt, get_text=True, is_GT=True)
    regions_hyp, pages_hyp = get_regions(args.path_page_hyp, is_GT=False)
    

    acts_gt  = create_acts_chancery(args.path_gt, args=args, is_GT=True)
    acts_hyp = create_acts_chancery(args.path_hyp, args=args, is_GT=False)

    IG_order = read_IG(args.IG_file, args.num_words)
    v_acts_gt = create_vector_acts(regions_gt, acts_gt, args.path_prix, IG_order=IG_order)
    v_acts_hyp = create_vector_acts(regions_hyp, acts_hyp, args.path_prix, IG_order=IG_order)
    total_RW_gt_all = np.sum(v_acts_gt)
    # print(len(v_acts_hyp), len(acts_hyp))
    # print(len(v_acts_gt), len(acts_gt))

    print(f"total_RW_gt {total_RW_gt_all}")
    print(f"Number of HYP acts {len(v_acts_hyp)} vs number of GT acts {len(v_acts_gt)}")

    groups_acts_v = restore_groups(v_acts_hyp, acts_hyp)
    groups_acts_v_gt = restore_groups(v_acts_gt, acts_gt)
    res = []
    baers = []
    for group_name, groups in groups_acts_v.items():
        v_group = [x[1] for x in groups]
        # print(groups)
        v_acts_gt_group = groups_acts_v_gt[group_name]
        v_group_gt = [x[1] for x in v_acts_gt_group]
        total_RW_gt = np.sum(v_group_gt)
        print(f" {group_name} RW = {total_RW_gt}")
        # print(v_acts_gt_group)
        cost, edits = levenshtein(v_group, v_group_gt, cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
        # all_edits.extend(edits)
        res.append(f"\n\n -- {group_name}")
        # exit()
        # print(edits)
        # print(f"Total cost: {cost} -- > {cost/total_RW_gt}" )
        delete_cost, ins_cost, subs_cost = 0, 0, 0
        num_dels, num_ins, num_subs, num_match = 0, 0, 0, 0
        for edit in edits:
            cost = edit['cost']
            if edit['type'] == 'deletion':
                print(f"delete {edit} i {np.sum(v_acts_gt[edit['j']])} -> GT {acts_gt[edit['j']]}")
                # print(f"delete {edit} -> {acts_hyp[edit['i']]}")
                delete_cost += cost
                num_dels += 1
            elif edit['type'] == 'insertion':
                print(f"insertion {edit} {np.sum(v_acts_hyp[edit['i']])} -> HYP {acts_hyp[edit['i']]}")
                # print(f"insertion {edit} {acts_gt[edit['j']]}")
                ins_cost += cost
                num_ins += 1
            elif edit['type'] == 'substitution':
                # c, _ = levenshtein(v_acts_hyp[edit["i"]], v_acts_gt[edit["j"]], cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
                print(f"Subs {edit} GT {acts_gt[edit['j']]} -> HYP {acts_hyp[edit['i']]}")
                subs_cost += cost
                num_subs += 1
            else:
                num_match += 1
        weighted_cost = (delete_cost + ins_cost + subs_cost) / total_RW_gt
        baer = weighted_cost*100.0
        res.append(f'\n BAER cost : {baer:.2f}% ({(delete_cost + ins_cost + subs_cost):.2f} / {total_RW_gt:.2f}) -> \n [{num_dels}] dels at cost {delete_cost:.2f}  \n [{num_ins}] ins at cost {ins_cost:.2f} \n [{num_subs}] subs at cost  {subs_cost:.2f} \n [{num_match}] matches')
        baers.append(baer)
    for r in res:
        print(r)
    print(f" \n \n -- (Macro) Mean BAER: {np.mean(baers):.2f}%")

    # print(acts_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--corpus', type=str, help='path to save the save', default="JMBD")
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--IG_file', type=str, help='The span results file')
    parser.add_argument('--num_words', type=int, help='The span results file', default=16384)
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    parser.add_argument('--path_page_hyp', type=str, help='')
    parser.add_argument('--path_page_gt', type=str, help='')
    parser.add_argument('--text_hyp', type=str, default='no')
    parser.add_argument('--path_prix', type=str, help='path to save the save')
    parser.add_argument('--max_iou', type=float, help='no', default=0.3)
    args = parser.parse_args()
    # args.do_train = args.text_hyp.lower() in ["si", "true", "yes"]
    main(args)