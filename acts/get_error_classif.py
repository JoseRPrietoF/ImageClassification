### obtaining the results
import argparse, re, os, numpy as np
from sklearn.metrics import balanced_accuracy_score



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
        if "." in fname:
            fname = fname.split(".")[0]
        acts[fname] = c
    return acts

def shannon_entropy(probabilities):
    l = np.nan_to_num(np.log2(probabilities))
    return -np.sum(probabilities * l)
    
def cross_entropy(probabilities, onehot_gt):
    # oh * log(probs)
    l = np.nan_to_num(onehot_gt * np.log2(probabilities))
    return -np.sum(probabilities * l)

def get_results_hyp(path_file_hyp:str, res_gt:dict, results="results",  dict_classes=None, dict_classes_rev=None, from_decoder=False, args=None):
    f = open(path_file_hyp, "r")
    lines = f.readlines()[1:]
    f.close()
    errors = []
    uniques = list(zip(*np.unique([v for k,v in res_gt.items()], return_counts=True)))
    uniques = {k:v for k,v in uniques}
    classes = {k:0 for k,v in uniques.items()}
    entropies, cross_entropies = [], []
    entropies_class, cross_entropies_class = {v:[] for k,v in dict_classes.items()}, {v:[] for k,v in dict_classes.items()}
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        probs = [float(x) for x in line[2:]]
        
        if results == "results":
            if not from_decoder:
                c = dict_classes[np.argmax(probs)]
        probs = np.clip(probs, 1e-10, 1.0)
        se = shannon_entropy(probs)
        entropies.append(se)
        onehot_gt = np.zeros(len(dict_classes))
        onehot_gt[np.argmax(probs)] = 1.0
        cr = cross_entropy(probs, onehot_gt)
        cross_entropies.append(cr)
        entropies_class[c].append(se)
        cross_entropies_class[c].append(cr)

        c_gt = res_gt.get(fname, "M")
        # print(fname, probs, c, c_gt)
        errors.append(c!=c_gt)
        classes[c_gt] += c!=c_gt
    #     if c!=c_gt:
    #         print(f"{fname}, HYP {c} {probs[dict_classes_rev[c]]:.3f},  GT {c_gt}")
    # print(classes)
    if not args.only:
        print(f"Shannon Entropy {np.mean(entropies):.4f}")
        print(f"Shannon Entropy per class")
        for k,v in entropies_class.items():
            print(f"  --> Shannon Entropy class {k} {np.mean(v):.4f}")
        print(f"  --> Shannon Entropy mean {np.mean([np.mean(v) for k,v in entropies_class.items() ]):.4f}")
        print(f"Cross Entropy {np.mean(cross_entropies):.4f}")
        print(f"Cross Entropy per class")
        for k,v in cross_entropies_class.items():
            print(f"  --> Cross Entropy class {k} {np.mean(v):.4f}")
        print(f"  --> Cross Entropy mean {np.mean([np.mean(v) for k,v in cross_entropies_class.items() ]):.4f}")
    elif args.only == "cr":
        print(f"{np.mean(cross_entropies):.4f} & ", end="")
    elif args.only == "macro_cr":
        print(f"{np.mean([np.mean(v) for k,v in cross_entropies_class.items() ]):.4f} & ", end="")
    elif args.only == "imfcrmcr":
        print(f"{np.mean(cross_entropies_class['I']):.4f} & {np.mean(cross_entropies_class['M']):.4f} & {np.mean(cross_entropies_class['F']):.4f} & {np.mean(cross_entropies):.4f} & {np.mean([np.mean(v) for k,v in cross_entropies_class.items() ]):.4f}", end="\n")
    return errors, len(lines)

def get_results_hyp_weighted(path_file_hyp:str, res_gt:dict,  dict_classes=None, dict_classes_rev=None, from_decoder=False, args=None):
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
        if not from_decoder:
            c = dict_classes[np.argmax(probs)]
        c_gt = res_gt.get(fname, "M")
        # errors.append(c!=c_gt)
        classes[c_gt] += c!=c_gt
        res_hyp.append(c)
        res_gt_list.append(c_gt)
        
    # print(classes)
    classes2 = {k:(v/uniques[k]) for k,v in classes.items()}
    # print(classes2)
    # print(entropies)
    
    return np.sum([v for k,v in classes2.items()])/len(classes2), classes, uniques, res_hyp, res_gt_list

def main(args):
    acts_gt = get_gt(args.path_gt)
    
    num_pages = 0
    dict_classes = {
        0: "F",
        1: "I",
        2: "M"
    }

    dict_classes_rev = {v:k for k,v in dict_classes.items()}
    if "4946" in args.folders:
        num_pages += 1399
    if "4952" in args.folders:
        num_pages += 980
    if "4949" in args.folders:
        num_pages += 1615
    if "4952" in args.folders:
        num_pages += 1481
    if "sise" in args.folders.lower():
        num_pages += 121
        dict_classes = {
            0: "C",
            1: "F",
            2: "I",
            3: "M"
        }

        dict_classes_rev = {v:k for k,v in dict_classes.items()}
    
    errors_hyp, num_pages2 = get_results_hyp(args.path_hyp, acts_gt, args.results, dict_classes, dict_classes_rev, from_decoder=args.from_decoder, args=args)
    weighted, num_errors, uniques, res_hyp, res_gt = get_results_hyp_weighted(args.path_hyp, acts_gt,  dict_classes, dict_classes_rev, from_decoder=args.from_decoder, args=args)
    num_err = np.sum(errors_hyp)
    if not "sise" in args.folders.lower():
        # num_pages = np.sum([v for k,v in uniques.items()])
        num_pages = num_pages2
    if not args.only:
        print(f"{num_err} errors -> {(num_err /num_pages)*100.0:.2f}%")
        print(f"Weighted: {num_errors} errors from {uniques} uniques -> {(weighted)*100.0:.2f}%  - {np.sum([v for k,v in num_errors.items()])} errors")
        print("------------------------------\n")
    else:
        if args.only == "errors":
            print(f"{(num_err /num_pages)*100.0:.2f}  & {(weighted)*100.0:.2f}", end=" & ")
        elif args.only == "error":
            print(f"{(num_err /num_pages)*100.0:.2f}", end=" & ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--results', type=str, help='The span results file', default="results")
    parser.add_argument('--folders', type=str, help='The span results file', default="results")
    parser.add_argument('--from_decoder', type=str, help='The span results file', default="false")
    parser.add_argument('--only', type=str, help='The span results file', default="")
    args = parser.parse_args()
    args.from_decoder = args.from_decoder.lower() in ["si", "yes", "true"]
    main(args)