### obtaining the results
import argparse, re, os, sys, numpy as np
from PD import levenshtein
from get_results import sub_cost, read_IG, append_prix
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
from data.page import PAGE
from acts.sequences_SiSe_all_pages import get_xmls
from acts.get_results_groups_page import get_regions, append_text

def get_bbox(contours):
    xmax ,xmin = np.max([ x[0] for x in contours]), np.min([ x[0] for x in contours])
    ymax ,ymin = np.max([ x[1] for x in contours]), np.min([ x[1] for x in contours])
    return [xmin, ymin, xmax, ymax]

def create_vector_acts(acts:list, path_prix:str, IG_order:dict, regions_gt={}, replace_text=False):
    # legajo = acts[0][0].split("_")[1]
    res_acts = []
    for act in acts:
        vector = [0]*(len(IG_order)+1)
        for page in act:
            page_path_prix = os.path.join(path_prix, f"{page}.idx")
            # print(page_path_prix)
            page_name = "_".join(page.split("_")[:-1])
            if replace_text:
                # pname
                pname = "_".join(page.split("_")[:-1])
                coord_reg, info_Reg = regions_gt[pname]
                pname = "_".join(pname.split("_")[:-1])
                page_obj_hyp_text = replace_text[pname]
                textlines = page_obj_hyp_text.get_textLines(get_bbox(coord_reg), min=args.max_iou)
                text = '\n'.join([x[1] for x in textlines]).upper()
            else:
                text = regions_gt[page_name][1]['text'].upper()

            vector = append_text(vector, text, IG_order)

            # print(vector)
            # exit()
        # print(vector[-1], np.sum(vector[:-1]))
        vector = vector[:-1] # last component is the "residual" or other words outside of IG
        res_acts.append(vector)
    return res_acts

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

def create_acts_SiSe(path_file:str, args=None):
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
        *name, npage, num, _ = fname.split("_")
        num = int(num.split(".")[0])
        name = "_".join(name)
        for i in range(0, 1000):
            name2 = f'{name}'
            pages = res_groups.get(name2, [])
            if len(pages) == 0 or pages[-1][0]+1 == num:
                pages.append((num, fname, c))
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

def main(args):
    regions_gt, pages_gt = get_regions(args.path_page_gt, get_text=True, is_GT=True)
    acts_gt  = create_acts_SiSe(args.path_gt, args=args)
    acts_hyp = create_acts_SiSe(args.path_hyp, args=args)
    IG_order = read_IG(args.IG_file, args.num_words)
    textlines_hyp = False
    if args.text_hyp != "no":
        textlines_hyp = load_textlines(args.text_hyp)
    v_acts_hyp = create_vector_acts(acts_hyp, args.path_page_gt, IG_order, regions_gt, replace_text=textlines_hyp)
    v_acts_gt = create_vector_acts(acts_gt, args.path_page_gt, IG_order, regions_gt)
    total_RW_gt = np.sum(v_acts_gt)
    # print(len(v_acts_hyp), len(acts_hyp))
    # print(len(v_acts_gt), len(acts_gt))
    print(f"total_RW_gt {total_RW_gt}")
    print(f"Number of HYP acts {len(v_acts_hyp)} vs number of GT acts {len(v_acts_gt)}")

    groups_acts_v = restore_groups(v_acts_hyp, acts_hyp, end_with_class=True)
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

    # cost, edits = levenshtein(v_acts_hyp, v_acts_gt, cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
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
    # parser.add_argument('--path_prix', type=str, help='path to save the save')
    parser.add_argument('--path_page_gt', type=str, help='')
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--IG_file', type=str, help='The span results file')
    parser.add_argument('--num_words', type=int, help='The span results file', default=16384)
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    parser.add_argument('--text_hyp', type=str, default='no')
    parser.add_argument('--max_iou', type=float, help='no', default=0.3)
    args = parser.parse_args()
    main(args)