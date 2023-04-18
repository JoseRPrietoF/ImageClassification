### obtaining the results
import argparse, re, os, numpy as np
from get_results import sub_cost, read_IG, create_vector_acts
from PD import levenshtein
from tabulate import tabulate

def get_name_acts(l, args, ):
    res = []
    if args.action in ["align"]:
        r = []
        for act in l:
            if act == ",":
                res.append(r)
                r = []
            else:
                r.append(f"JMBD_{args.folder}_{act.zfill(5)}")
        res.append(r)
        return res
    for act in l:
        res.append(f"JMBD_{args.folder}_{act.zfill(5)}")
    return res

def main(args):
    IG_order = read_IG(args.IG_file, args.num_words)
    files1 = get_name_acts(args.files1, args)
    if args.action in ["align"]:
        files1_v = create_vector_acts(files1, args.path_prix, IG_order)
        files2 = get_name_acts(args.files2, args)
        files2_v = create_vector_acts(files2, args.path_prix, IG_order)
    else:
        files1 = create_vector_acts([files1], args.path_prix, IG_order)[0]
    if args.action in ["del", "deletion", "delete", "insert", "ins", "insertion"]:
        cost = np.sum(files1)
        print(f"Cost of ins/del for {args.files1} in folder {args.folder} : {cost}")
    elif args.action in ["subs", "sub", "substitution"]:
        files2 = get_name_acts(args.files2, args)
        files2 = create_vector_acts([files2], args.path_prix, IG_order)[0]
        cost = sub_cost(files1, files2)
        print(f"Cost of substitution act {args.files1} for act {args.files2} in folder {args.folder} : {cost}")
    elif args.action in ["align"]:
        cost, edits, rows = levenshtein(files2_v, files1_v, cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum, return_rows=True)
        # print(cost)
        for edit in edits:
            if edit['type'] == 'deletion':
                print(f"delete {edit} i {np.sum(files2_v[edit['j']])} -> GT {files1[edit['j']]}")
                # print(f"delete {edit} -> {acts_hyp[edit['i']]}")

            elif edit['type'] == 'insertion':
                print(f"insertion {edit} {np.sum(files2_v[edit['i']])} -> HYP {files2[edit['i']]}")
                # print(f"insertion {edit} {acts_gt[edit['j']]}")

            elif edit['type'] == 'substitution':
                # c, _ = levenshtein(v_acts_hyp[edit["i"]], v_acts_gt[edit["j"]], cost_subs=sub_cost, del_cost=np.sum, ins_cost=np.sum)
                print(f"Subs {edit} GT {files1[edit['j']]} -> HYP {files2[edit['i']]}")
            else:
                print(edit)
        print("\n\n\n")
        headers = ["-",*files1, ]
        files2.insert(0,"-")
        rows = [[files2[i],*rows[i]] for i in range(len(rows))]
        rows = [ *rows]
        print(tabulate(rows,  headers=headers              ))
    else:
        raise Exception(f"Operation {args.action} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--folder', type=str, help='path to save the save')
    parser.add_argument('--path_prix', type=str, help='path to save the save')
    parser.add_argument('--IG_file', type=str, help='path to save the save')
    parser.add_argument('--action', type=str, help='path to save the save')
    parser.add_argument('--files1', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--files2', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--num_words', type=int, help='The span results file', default=16384)
    args = parser.parse_args()
    main(args)