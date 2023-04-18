import argparse, re, numpy as np

dict_ = {
    "F": 0,
    "I": 1, 
    "M": 2
}

rev_dict_ = {v:k for k,v in dict_.items()}

def count_inconsistencies(s:str, impossible_states:list=["II", "FF", "FM"]):
    total = 0
    for i in impossible_states:
        s_i = s.count(i)
        print(i, s_i)
        total += s_i
    total *= 2
    if not s.startswith("I"):
        print("I no is the first one")
        total += 1
    if not s.endswith("F"):
        print("F is not the last one")
        total += 1
    return total

def read_file(path_file:str):
    f = open(path_file, "r")
    lines = f.readlines()
    f.close()
    res = []
    for line in lines:
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = int(line[0].split("_")[-1]), line[1]
        res.append((fname, c))
    res.sort()
    res = "".join([x[1] for x in res])
    return res

def read_results(p:str):
    file = open(p, "r")
    res = []
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        fname, _, fprob, iprob, mprob = line.strip().split(" ")
        fprob, iprob, mprob = float(fprob), float(iprob), float(mprob)
        pnumber = int(fname.split("_")[-1])
        hyp_l = rev_dict_.get(np.argmax([fprob, iprob, mprob]))
        res.append([pnumber, hyp_l])
    res.sort()
    file.close()
    res = "".join([x[1] for x in res])
    return res

def main(args):
    # s = "IMFMFIF"
    # s = read_file(args.path)
    s = read_results(args.path)
    print(s)
    count = count_inconsistencies(s)
    if "4946" in args.path:
        num_pages = 1399
    elif "4952" in args.path:
        num_pages = 980
    print(f"{(count/len(s))*100.0:.2f}% [{count} incosistencies from {num_pages} pages]")  # Output: 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--path', type=str, help='model')
    args = parser.parse_args()
    main(args)