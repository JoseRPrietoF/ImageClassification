import numpy as np, argparse
import glob, os, sys
from math import log as log_
from PD import greedy
from sequences_SiSe import get_transition_probs, log, PD_prob_trans, calc_prob_transitions, cut_segments, get_imgs, raw_process
DOSSIER_COURRANT = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PARENT = os.path.dirname(DOSSIER_COURRANT)
sys.path.append(DOSSIER_PARENT)
sys.path.append(os.path.dirname(DOSSIER_PARENT))
from data.page import PAGE

dict_ = {
    "C": 0,
    "F": 1,
    "I": 2, 
    "M": 3
}

rev_dict_ = {v:k for k,v in dict_.items()}

posibles_states = [("I", "M"), ("I", "F"), ("M", "M"), ("M", "F"), ("F", "I")]

MINVALUE = np.finfo(float).eps

def read_results(p:str):
    file = open(p, "r")
    res = []
    res_dict = {}
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        fname, _, cprob, fprob, iprob, mprob = line.strip().split(" ")
        fname = fname.split(".")[0]
        cprob, fprob, iprob, mprob = float(cprob), float(fprob), float(iprob), float(mprob)
        # fprob = max(0.0001, fprob); iprob = max(0.0001, iprob); mprob = max(0.0001, mprob); 
        pnumber = "_".join(fname.split("_")[:-1])
        res.append([pnumber, fname, cprob, fprob, iprob, mprob])
        hyp_l = rev_dict_.get(np.argmax([cprob, fprob, iprob, mprob]))
        res_dict[fname] = [cprob, fprob, iprob, mprob]
    res.sort()
    file.close()
    res_groups = {}
    for pnumber, fname, cprob, fprob, iprob, mprob in res:
        *name, npage, num, c = fname.split("_")
        num = int(num.split(".")[0])
        name = "_".join(name)
        for i in range(0, 1000):
            name2 = f'{name}'
            pages = res_groups.get(name2, [])
            if len(pages) == 0 or pages[-1][0]+1 == num:
                pages.append((num, fname, cprob, fprob, iprob, mprob))
                break
        res_groups[name2] = pages
    return res_groups, res_dict

def get_xmls(p:str):
    xmls = glob.glob(os.path.join(p, "*xml"))
    xmls.sort()
    return xmls

def sort_regions(acts):
    # For double page at max
    regions = []
    regions2 = []
    for acti, act in enumerate(acts):
        coords = act[0]
        min_x = np.min(coords[:,0])
        min_y = np.min(coords[:,1])
        info = [[min_y, min_x], acti, act]
        regions.append(info)
    regions = sorted(regions, key = lambda x: x[0])
    res = [act for _,_,act in regions]
    return res

def get_regions(args, max_iou_acts = 0.99, from_classif=None):
    offset_acts = 0
    hyp_acts = []
    xmls = get_xmls(args.path_page_hyp)
    for xml in xmls:
        page = PAGE(xml)
        acts = page.get_textRegionsActs(max_iou=max_iou_acts, GT=False)
        acts = sort_regions(acts)
        for coords, id_act, info in acts:
            hyp_acts.append((id_act, info["type"], info["probs"], coords))
    res_groups = {}
    for id_act, _, info, coords in hyp_acts:
        name = "_".join(id_act.split("_")[:2])
        if from_classif is not None:
            cprob, fprob, iprob, mprob = from_classif[id_act]
        else:
            cprob, fprob, iprob, mprob = info['AC'], info['AF'], info['AI'], info['AM']
       
        pages = res_groups.get(name, [])
        pages.append((id_act, id_act, cprob, fprob, iprob, mprob, coords))
        res_groups[name] = pages
    for k, v in res_groups.items():
        v.sort()
        per_pages = {}
        for id_act, id_act, cprob, fprob, iprob, mprob, coords in v:
            name_pag = "_".join(id_act.split("_")[:-1])
            coord_top_y = np.min(coords[:,1])
            pages = per_pages.get(name_pag, [])
            pages.append([coord_top_y, id_act, id_act, cprob, fprob, iprob, mprob])
            per_pages[name_pag] = pages
        for k2,v2 in per_pages.items():
            v2.sort()
        pages = list(per_pages.keys())
        pages.sort()
        res = []
        for pages_acts in pages:
            for coord_top_y, id_act, id_act, cprob, fprob, iprob, mprob in per_pages[pages_acts]:
                res.append([id_act, id_act, cprob, fprob, iprob, mprob])
        res_groups[k] = res
    return res_groups

def get_transition_probs(GT_path):
    f = open(GT_path, "r")
    lines = f.readlines()
    f.close()
    sequence = " ".join([line.strip().split(" ")[1] for line in lines])
    im_c = sequence.count("AI AM")
    if_c = sequence.count("AI AF")
    mm_c = sequence.count("AM AM")
    mf_c = sequence.count("AM AF")
    fc_c = sequence.count("AF AC")
    fi_c = sequence.count("AF AI")
    cc_c = sequence.count("AC AC")
    ci_c = sequence.count("AC AI")
    total = im_c + if_c + mm_c + mf_c + fc_c + cc_c + ci_c
    if_ = if_c / (if_c + im_c)
    im = im_c / (if_c + im_c)
    try:
        mm = mm_c / (mf_c + mm_c)
        mf = mf_c / (mf_c + mm_c)
    except:
        mm = 0
        mf = 0
    fi = fi_c / (fi_c + fc_c)
    fc = fc_c / (fi_c + fc_c)
    # Cs
    cc = cc_c /  (cc_c + ci_c)
    ci = ci_c /  (cc_c + ci_c)
    m = {}
    m[("I","M")]=log(im)
    m[("I","F")]=log(if_)
    m[("M","M")]=log(mm)
    m[("M","F")]=log(mf)
    m[("F","I")]=log(fi)
    m[("F","C")]=log(fc)
    m[("C","C")]=log(cc)
    m[("C","I")]=log(ci)
    return m
    
def main(args, GT:str):
    from_classif = None
    if args.alg == "PD":
        m = get_transition_probs(args.GT)
    # results, res_dict = read_results(p)
    results = get_regions(args, from_classif=from_classif)
    res = []
    names = []
    c = 0
    for name_group, results_group in results.items():
        c += len(results_group)
        for r in results_group:
            names.append([*r[2:], r[1]])
        if args.alg == "PD":
            res2 = PD_prob_trans(results_group, m, open_group=False)
        elif args.alg == "greedy":
            res2 = greedy(results_group)
        elif args.alg == "raw":
            res2 = raw_process(results_group)
        res.extend(res2)
    
    num_exps_pd = cut_segments(res, assum_consistency=True)
    # print("-----------------------------")
    # print("FNAME \t C \t probFinal \t probInitial \t probMiddle")
    for img, c in zip(names, res):
        print(img[-1], "\t ", "\t ", c, f" \t {img[0]} \t {img[1]} \t {img[2]} \t {img[3]}")

    # print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--GT', type=str, help='algorithm', default="")
    parser.add_argument('--path_page_hyp', type=str, help='')
    parser.add_argument('--from_classif', type=str, help='', default="")
    # parser.add_argument('--path_page_gt', type=str, help='')
    # parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    args = parser.parse_args()
    main(args, GT=args.GT)