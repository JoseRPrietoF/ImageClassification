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

def create_vector_acts(regions:list, acts:list, pages:list, IG_order:dict, args, pages_gt:list=None, is_GT=False, replace_text=False):
    res_acts = []
    for act in acts:
        vector = [0]*(len(IG_order)+1)
        for act_region in act:
            if pages_gt is not None:
                pname = "_".join(act_region.split("_")[:-1])
                page_obj = pages_gt[pname]
            else:
                act_region = "_".join(act_region.split("_")[:-1])
                pname = "_".join(act_region.split("_")[:-1])
                page_obj = pages[pname]
            coord_reg, info_Reg = regions[act_region]
            if replace_text:
                # pname
                page_obj_hyp_text = replace_text[pname]
                textlines = page_obj_hyp_text.get_textLines(get_bbox(coord_reg), min=args.max_iou)
                text = '\n'.join([x[1] for x in textlines]).upper()
            else:
                # coord_reg, info_Reg = regions[act_region]
                if is_GT:
                    text = info_Reg['text'].upper()
                else:
                    textlines = page_obj.get_textLines(get_bbox(coord_reg), min=args.max_iou)
                    text = '\n'.join([x[1] for x in textlines]).upper()
            # print(page_path_prix)
            vector = append_text(vector, text, IG_order)
        # print(vector[-1], np.sum(vector[:-1]))
        vector = vector[:-1] # last component is the "residual" or other words outside of IG
        res_acts.append(vector)
    return res_acts

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
            res[id_act] = (coords, info)
    return res, pages

def create_acts_SiSe(path_file:str, texts_gt=None, args=None, is_GT=False):
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
        if is_GT:
            *name, npage, num, _ = fname.split("_")
        else:
            *name, npage, num = fname.split("_")
        num = int(num.split(".")[0])
        npage = int(npage)
        name = "_".join(name)
        for i in range(0, 1000):
            name2 = f'{name}'
            pages = res_groups.get(name2, [])
            if len(pages) == 0 or pages[-1][0]+1 == npage or pages[-1][0] == npage:
                pages.append((npage, fname, c))
                break
        res_groups[name2] = pages
    
    acts = []
    for name_group, results_group in res_groups.items(): 
        aux = []
        for num, fname, c in results_group:
            aux.append(fname)
            if args.alg == "raw":
                if args.class_to_cut == c:
                    acts.append(aux)
                    aux = []
            else:
                if c in ["F", "C"]:
                    acts.append(aux)
                    aux = []
        if aux:
            acts.append(aux)
    return acts

def load_textlines(text_hyp_paths):
    # get_textLines_normal
    xmls = get_xmls(text_hyp_paths)
    res = {}
    # exit()
    for xml in xmls:
        name = xml.split("/")[-1].split(".")[0]
        page = PAGE(xml)
        # tls = page.get_textLines_normal()
        res[name] = page
    return res

def restore_groups(v_acts, acts, end_with_class=False):
    res = {}
    for group, v_group in zip(acts, v_acts):
        if end_with_class:
            group_name = "_".join(group[0].split("_")[:-3])
        else:
            group_name = "_".join(group[0].split("_")[:-2])
        list_g = res.get(group_name, [])
        list_g.append((group, v_group))
        res[group_name] = list_g
    return res

def main(args):
    regions_hyp, pages_hyp = get_regions(args.path_page_hyp, is_GT=False)
    regions_gt, pages_gt = get_regions(args.path_page_gt, get_text=True, is_GT=True)
    acts_gt  = create_acts_SiSe(args.path_gt, args=args, is_GT=True)
    acts_hyp = create_acts_SiSe(args.path_hyp, texts_gt=args.path_gt, args=args, is_GT=False)

    IG_order = read_IG(args.IG_file, args.num_words)
    textlines_hyp = False
    if args.text_hyp != "no":
        textlines_hyp = load_textlines(args.text_hyp)
    v_acts_hyp = create_vector_acts(regions_hyp, acts_hyp, pages_hyp, pages_gt=pages_gt, IG_order=IG_order, args=args, replace_text=textlines_hyp)
    v_acts_gt = create_vector_acts(regions_gt, acts_gt, pages_gt, IG_order=IG_order, args=args, is_GT=True)
    total_RW_gt = np.sum(v_acts_gt)
    # print(len(v_acts_hyp), len(acts_hyp))
    # print(len(v_acts_gt), len(acts_gt))

    print(f"total_RW_gt {total_RW_gt}")
    print(f"Number of HYP acts {len(v_acts_hyp)} vs number of GT acts {len(v_acts_gt)}")

    groups_acts_v = restore_groups(v_acts_hyp, acts_hyp)
    groups_acts_v_gt = restore_groups(v_acts_gt, acts_gt, end_with_class=True)
    all_edits = []
    for group_name, groups in groups_acts_v.items():
        v_group = [x[1] for x in groups]
        # print(groups)
        
        v_acts_gt_group = groups_acts_v_gt[group_name]
        v_group_gt = [x[1] for x in v_acts_gt_group]
        # print(v_acts_gt_group)
        cost, edits = levenshtein(v_group, v_group_gt, cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
        all_edits.extend(edits)
    # exit()
    # print(edits)
    # print(f"Total cost: {cost} -- > {cost/total_RW_gt}" )
    delete_cost, ins_cost, subs_cost = 0, 0, 0
    num_dels, num_ins, num_subs, num_match = 0, 0, 0, 0
    for edit in all_edits:
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
    print(f'\n Act BoWWER cost : {weighted_cost*100.0:.2f}% ({(delete_cost + ins_cost + subs_cost)} / {total_RW_gt}) -> \n [{num_dels}] dels at cost {delete_cost}  \n [{num_ins}] ins at cost {ins_cost} \n [{num_subs}] subs at cost  {subs_cost} \n [{num_match}] matches')
    print(f"GT {len(v_acts_gt)} = {num_match+num_subs+num_dels} : {len(v_acts_gt) == num_match+num_subs+num_dels}")
    print(f"HYP {len(v_acts_hyp)} = {num_match+num_subs+num_ins} : {len(v_acts_hyp) == num_match+num_subs+num_ins}")

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
    parser.add_argument('--max_iou', type=float, help='no', default=0.3)
    args = parser.parse_args()
    # args.do_train = args.text_hyp.lower() in ["si", "true", "yes"]
    main(args)