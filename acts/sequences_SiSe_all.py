import numpy as np, argparse
import glob, os
from math import log as log_
from PD import greedy
from sequences_SiSe import get_transition_probs, log, PD_prob_trans, calc_prob_transitions, cut_segments, get_imgs, raw_process, get_transition_probs

posibles_states = [("I", "M"), ("I", "F"), ("M", "M"), ("M", "F"), ("F", "I")]

MINVALUE = np.finfo(float).eps
DEBUG_GT = False

def read_results(p:str):
    file = open(p, "r")
    res = []
    res_dict = {}
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        fname, c, cprob, fprob, iprob, mprob = line.strip().split(" ")
        fname = fname.split(".")[0]
        cprob, fprob, iprob, mprob = float(cprob), float(fprob), float(iprob), float(mprob)
        if DEBUG_GT:
            cprob, fprob, iprob, mprob = 0,0,0,0
            if c == "F":
                fprob = 1.0
            elif c == "I":
                iprob = 1.0
            elif c == "M":
                mprob = 1.0
            else:
                cprob = 1.0
        # fprob = max(0.0001, fprob); iprob = max(0.0001, iprob); mprob = max(0.0001, mprob); 
        pnumber = "_".join(fname.split("_")[:-1])
        res.append([pnumber, fname, cprob, fprob, iprob, mprob])
        res_dict[fname] = [cprob, fprob, iprob, mprob]
    res.sort()
    file.close()
    res_groups = {}
    for pnumber, fname, cprob, fprob, iprob, mprob in res:
        *name, npage, num, _ = fname.split("_")
        num = int(num.split(".")[0])
        npage = int(npage)
        name = "_".join(name)
        
        for i in range(0, 1000):
            name2 = f'{name}'
            pages = res_groups.get(name2, [])
            if len(pages) == 0 or pages[-1][0]+1 == npage or pages[-1][0] == npage:
                pages.append((npage, fname, cprob, fprob, iprob, mprob))
                break
        res_groups[name2] = pages
    return res_groups, res_dict

def main(p:str, sorted_imgs:list, alg:str, GT:str):
    if alg == "PD":
        m = get_transition_probs(GT)
    results, res_dict = read_results(p)
    res = []
    for name_group, results_group in results.items():
        if alg == "PD":
            res2 = PD_prob_trans(results_group, m, open_group=True)
        elif alg == "greedy":
            res2 = greedy(results_group)
        elif alg == "raw":
            res2 = raw_process(results_group)
        res.extend(res2)
    # num_exps_pd = cut_segments(res, assum_consistency=True)
    # print(f'{num_exps_pd} expedients')
    # print("-----------------------------")
    # print("FNAME \t C \t probFinal \t probInitial \t probMiddle")
    for img, c in zip(sorted_imgs, res):
        print(img, "\t ", "\t ", c, f" \t {res_dict[img][0]} \t {res_dict[img][1]} \t {res_dict[img][2]} \t {res_dict[img][3]}")

    # print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--folder', type=str, help='folder')
    parser.add_argument('--folder_orig', type=str, help='folder', default="")
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--GT', type=str, help='algorithm', default="")
    parser.add_argument('--path_res', type=str, help='algorithm', default="")
    parser.add_argument('--path_imgs', type=str, help='algorithm', default="")
    # parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    args = parser.parse_args()
    # prod = "4946"
    # model = "resnet18"
    # alg = "greedy"
    alg = args.alg
    prod = args.folder
    model = args.model
    folder_orig = args.folder_orig
    if folder_orig == "":
        folder_orig = prod
    # print(f'Prod for folder {prod}')
    path_res = args.path_res
    path_imgs = args.path_imgs
    main(path_res, get_imgs(path_imgs), alg=alg, GT=args.GT)