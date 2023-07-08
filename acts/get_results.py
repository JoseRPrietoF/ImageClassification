### obtaining the results
import argparse, re, os, numpy as np
from PD import levenshtein

def check_consistency(res:list):
    pass

def create_acts_SiSe(path_file:str, used_pages_gt:set=None, args=None):
    f = open(path_file, "r")
    lines = f.readlines()
    f.close()
    acts = []
    aux = []
    used_pages = set()
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        num_page = "_".join(fname.split("_")[:-1])
        # aux.append((num_page, fname))
        fname = fname.split(".")[0]
        aux.append(fname)
        if args.alg == "raw":
            if args.class_to_cut == c:
                acts.append(aux)
                aux = []
        else:
            if c in ["F", "C"]:
                acts.append(aux)
                aux = []
    return acts

def create_acts(path_file:str, used_pages_gt:set=None, args=None):
    f = open(path_file, "r")
    lines = f.readlines()
    f.close()
    acts = []
    aux = []
    used_pages = set()
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        num_page = int(fname.split("_")[-1])
        aux.append((num_page, fname))
        if args.alg == "raw":
            if args.class_to_cut == c:
                acts.append(aux)
                aux = []
        else:
            if c == "F":
                acts.append(aux)
                aux = []
    legajo = fname.split("_")[1]
    new_acts = []
    new_acts_npages = []
    # Append white pages in the medium of the acts
    for act in acts:
        ini, fin = act[0][0], act[-1][0]
        new_act = []
        new_act_npages = []
        for i in range(ini, fin+1):
            fname = f"JMBD_{legajo}_{i:05}"
            if used_pages_gt is not None and fname in used_pages_gt: # HYP
                new_act.append(fname)
                new_act_npages.append(i)
                used_pages_gt.remove(fname)
            elif used_pages_gt is None: # GT
                new_act.append(fname)
            else:
                print(f"Skipped {fname} page in hyp (not in GT but found in HYP)")
            used_pages.add(fname)
        new_acts.append(new_act)
        new_acts_npages.append(new_act_npages)
    # Time to add pages from GT but not in HYP
    if used_pages_gt is not None:
        for page_Gt in used_pages_gt:
            npage_gt_tosearch = int(page_Gt.split("_")[-1]) - 1 #we are looking for the previous one to add the actual page 
            for i, (act, num_p_act) in enumerate(zip(new_acts, new_acts_npages)):
                if npage_gt_tosearch in num_p_act:
                    new_acts[i].append(page_Gt)
                    break
    return new_acts, used_pages

def read_IG(IG_file:str, num_words:int):
    f = open(IG_file, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        word = line.split(" ")[0].upper()
        res.append(word)
    res = res[:num_words]
    res_dict = {}
    for i, word in enumerate(res):
        res_dict[word] = i
    return res_dict

def append_prix(vector:list, page_path_prix:list, IG_order:dict):
    with open(page_path_prix, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            word, prob = line[0].upper(), float(line[2])
            pos_word = IG_order.get(word, -1)
            # if pos_word is not None:
            vector[pos_word] += prob
    return vector

def create_vector_acts(acts:list, path_prix:str, IG_order:dict):
    # legajo = acts[0][0].split("_")[1]
    res_acts = []
    for act in acts:
        vector = [0]*(len(IG_order)+1)
        for page in act:
            page_path_prix = os.path.join(path_prix, f"{page}.idx")
            # print(page_path_prix)
            vector = append_prix(vector, page_path_prix, IG_order)
            # print(vector)
            # exit()
        # print(vector[-1], np.sum(vector[:-1]))
        vector = vector[:-1] # last component is the "residual" or other words outside of IG
        res_acts.append(vector)
    return res_acts

def sub_cost(a, b):
    # Following https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4355392 
    # Eq. (6) without norm
    a = np.array(a)
    b = np.array(b)
    diff = np.sum(np.absolute(np.array(a) - np.array(b)))
    Na = np.sum(a)
    Nb = np.sum(b)
    return (np.abs(Na-Nb) + diff) / 2

def main(args):
    if "SiSe" in args.corpus:
        acts_gt  = create_acts_SiSe(args.path_gt, args=args)
        acts_hyp = create_acts_SiSe(args.path_hyp, args=args)
    else:
        acts_gt, used_pages_gt = create_acts(args.path_gt, used_pages_gt=None, args=args)
        acts_hyp, _ = create_acts(args.path_hyp, used_pages_gt=used_pages_gt, args=args)
    IG_order = read_IG(args.IG_file, args.num_words)
    # v_acts_hyp = create_vector_acts([acts_hyp[2]], args.path_prix, IG_order)
    # v_acts_gt = create_vector_acts([acts_gt[2]], args.path_prix, IG_order)   
    v_acts_hyp = create_vector_acts(acts_hyp, args.path_prix, IG_order)
    v_acts_gt = create_vector_acts(acts_gt, args.path_prix, IG_order)
    
    total_RW_gt = np.sum(v_acts_gt)
    # print(len(v_acts_hyp), len(acts_hyp))
    # print(len(v_acts_gt), len(acts_gt))
    print(f"total_RW_gt {total_RW_gt}")
    print(f"Number of HYP acts {len(v_acts_hyp)} vs number of GT acts {len(v_acts_gt)}")

    cost, edits = levenshtein(v_acts_hyp, v_acts_gt, cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
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
    print(f'\n Act BoWWER cost : {weighted_cost*100.0:.2f}% ({(delete_cost + ins_cost + subs_cost)} / {total_RW_gt}) -> \n [{num_dels}] dels at cost {delete_cost}  \n [{num_ins}] ins at cost {ins_cost} \n [{num_subs}] subs at cost  {subs_cost} \n [{num_match}] matches')
    print(f"GT {len(v_acts_gt)} = {num_match+num_subs+num_dels} : {len(v_acts_gt) == num_match+num_subs+num_dels}")
    print(f"HYP {len(v_acts_hyp)} = {num_match+num_subs+num_ins} : {len(v_acts_hyp) == num_match+num_subs+num_ins}")

    
    # print(acts_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--corpus', type=str, help='path to save the save', default="JMBD")
    parser.add_argument('--path_prix', type=str, help='path to save the save')
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--IG_file', type=str, help='The span results file')
    parser.add_argument('--num_words', type=int, help='The span results file', default=16384)
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    args = parser.parse_args()
    main(args)