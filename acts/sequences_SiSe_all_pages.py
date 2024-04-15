import numpy as np, argparse
import glob, os, sys
from math import log as log_
from PD import greedy
from sequences_SiSe import get_transition_probs, log, PD_prob_trans, calc_prob_transitions, cut_segments, get_imgs, raw_process, get_transition_probs
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
            hyp_acts.append((id_act, info["type"], info["probs"]))
    # print(len(hyp_acts))
    # exit()
    res_groups = {}
    for id_act, _, info in hyp_acts:
        *name, npage, num = id_act.split("_")
        num = int(num.split(".")[0])
        npage = int(npage)
        name = "_".join(name)
        if from_classif is not None:
            cprob, fprob, iprob, mprob = from_classif[id_act]
        else:
            cprob, fprob, iprob, mprob = info['AC'], info['AF'], info['AI'], info['AM']
        for i in range(0, 10):
            name2 = f'{name}'
            pages = res_groups.get(name2, [])

            if len(pages) == 0 or pages[-1][0]+1 == npage or pages[-1][0] == npage:
                pages.append((npage, id_act, cprob, fprob, iprob, mprob))
                break

        res_groups[name2] = pages
    return res_groups

def load_classif(path:str):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines[1:]:
        name, _, *probs = line.strip().split(" ")
        probs = [float(f) for f in probs]
        name = "_".join(name.split("_")[:-1])
        res[name] = probs
    return res

def main(args, GT:str):
    from_classif = None
    if args.from_classif != "":
        from_classif = load_classif(args.from_classif)
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
            res2 = PD_prob_trans(results_group, m, open_group=True)
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
    parser.add_argument('--folder', type=str, help='folder')
    parser.add_argument('--folder_orig', type=str, help='folder', default="")
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--GT', type=str, help='algorithm', default="")
    parser.add_argument('--path_page_hyp', type=str, help='')
    parser.add_argument('--from_classif', type=str, help='', default="")
    # parser.add_argument('--path_page_gt', type=str, help='')
    # parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    args = parser.parse_args()
    main(args, GT=args.GT)