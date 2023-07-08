### obtaining the results
import argparse, re, os, numpy as np
from PD import levenshtein
from get_results import sub_cost, read_IG, append_prix, create_vector_acts

def create_acts_SiSe(path_file:str, used_pages_gt:set=None, args=None):
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

def main(args):
    
    acts_gt  = create_acts_SiSe(args.path_gt, args=args)
    acts_hyp = create_acts_SiSe(args.path_hyp, args=args)
    IG_order = read_IG(args.IG_file, args.num_words)
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