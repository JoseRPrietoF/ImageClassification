### obtaining the results
import argparse, re, os, numpy as np
from sklearn.metrics import balanced_accuracy_score

dict_classes = {
    0: "F",
    1: "I",
    2: "M"
}

dict_classes_rev = {v:k for k,v in dict_classes.items()}

def get_gt(path_file_gt:str):
    f = open(path_file_gt, "r")
    lines = f.readlines()
    f.close()
    acts = {}
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        acts[fname] = c
    return acts

def get_results_hyp(path_file_hyp:str, res_gt:dict, results="results"):
    f = open(path_file_hyp, "r")
    lines = f.readlines()[1:]
    f.close()
    errors = []
    uniques = list(zip(*np.unique([v for k,v in res_gt.items()], return_counts=True)))
    uniques = {k:v for k,v in uniques}
    classes = {k:0 for k,v in uniques.items()}
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        probs = [float(x) for x in line[2:]]
        if results == "results":
            c = dict_classes[np.argmax(probs)]
        c_gt = res_gt.get(fname, "M")
        errors.append(c!=c_gt)
        classes[c_gt] += c!=c_gt
    #     if c!=c_gt:
    #         print(f"{fname}, HYP {c} {probs[dict_classes_rev[c]]:.3f},  GT {c_gt}")
    # print(classes)
    return errors

def get_results_hyp_weighted(path_file_hyp:str, res_gt:dict):
    f = open(path_file_hyp, "r")
    lines = f.readlines()[1:]
    f.close()
    # print(f"{len(lines)} lines")
    # print(f"{len(res_gt)} lines GT")
    errors = []
    uniques = list(zip(*np.unique([v for k,v in res_gt.items()], return_counts=True)))
    # print({k:v/len(res_gt) for k,v in uniques})
    uniques = {k:v for k,v in uniques}
    classes = {k:0 for k,v in uniques.items()}
    res_hyp, res_gt_list = [], []
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        probs = [float(x) for x in line[2:]]
        c = dict_classes[np.argmax(probs)]
        c_gt = res_gt.get(fname, "M")
        # errors.append(c!=c_gt)
        classes[c_gt] += c!=c_gt
        res_hyp.append(c)
        res_gt_list.append(c_gt)
    # print(classes)
    classes2 = {k:(v/uniques[k]) for k,v in classes.items()}
    # print(classes2)
    return np.sum([v for k,v in classes2.items()])/len(classes2), classes, uniques, res_hyp, res_gt_list

def main(args):
    acts_gt = get_gt(args.path_gt)
    errors_hyp = get_results_hyp(args.path_hyp, acts_gt, args.results)
    weighted, num_errors, uniques, res_hyp, res_gt = get_results_hyp_weighted(args.path_hyp, acts_gt)
    num_err = np.sum(errors_hyp)
    if "4946" in args.path_gt:
        num_pages = 1399
    elif "4952" in args.path_gt:
        num_pages = 980
    elif "4949" in args.path_gt:
        num_pages = 1615
    elif "4952" in args.path_gt:
        num_pages = 1481
    print(f"{num_err} errors -> {(num_err /num_pages)*100.0:.2f}%")
    print(f"Weighted: {num_errors} errors from {uniques} uniques -> {(weighted)*100.0:.2f}%  - {np.sum([v for k,v in num_errors.items()])} errors")
    # balanced_acc = balanced_accuracy_score(res_hyp, res_gt, adjusted=True)
    # print(f"balanced_acc {1-balanced_acc}")

    
    # print(acts_gt)
    print("------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--results', type=str, help='The span results file', default="results")
    args = parser.parse_args()
    main(args)