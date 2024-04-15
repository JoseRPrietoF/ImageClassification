import numpy as np, argparse
import glob, os
from math import log as log_
from PD import greedy
dict_ = {
    "F": 0,
    "I": 1, 
    "M": 2
}

rev_dict_ = {v:k for k,v in dict_.items()}

posibles_states = [("I", "M"), ("I", "F"), ("M", "M"), ("M", "F"), ("F", "I")]

MINVALUE = np.finfo(float).eps

def log(x):
    return x
    if x > 0: return log_(x)
    else:
        # print(x) 
        return -10000.841361487904734
        # return -np.inf

def read_results(p:str):
    file = open(p, "r")
    res = []
    res_dict = {}
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        # print(line.strip().split(" "))
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
    order = [i[2] for i in res]
    return res, order, res_dict

def read_results_text(p:str):
    file = open(p, "r")
    res = {}
    res_arr = []
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        fname, gt, iprob, mprob, fprob = line.strip().split(" ")
        fprob, iprob, mprob = float(fprob), float(iprob), float(mprob)
        # fprob = max(MINVALUE, fprob); iprob = max(MINVALUE, iprob); mprob = max(MINVALUE, mprob); 
        # print(fprob, iprob, mprob )
        pnumber = int(fname.split("_")[-1])
        res_arr.append([pnumber, fname, gt, fprob, iprob, mprob])
        # print(fname,  fprob, iprob, mprob)
        res[fname] = (gt, fprob, iprob, mprob)
    # print(f'{error} errors')
    res_arr.sort()
    file.close()
    # order = [i[2] for i in res]
    return res_arr, res

def PD_prob_trans(results, transProb:dict, res_text:dict=None, alpha=1.0, open_group=False):
    """
    transProb es el dict de probabilidades de transicion
    """
    npages = len(results)
    C,F,M,I = 0,1,2,3
    m = [[0]*npages, [0]*npages, [0]*npages, [0]*npages] # F M I 
    for i, (pnumber, fname, cprob, fprob, iprob, mprob) in enumerate(results):
        m[C][i] = log(cprob) * alpha
        m[F][i] = log(fprob) * alpha
        m[M][i] = log(mprob) * alpha
        m[I][i] = log(iprob) * alpha
        if res_text is not None:
            _,cprob,fprob,iprob,mprob = res_text[fname]     
            m[C][i] += log(cprob) * ( 1.0 - alpha)   
            m[F][i] += log(fprob) * ( 1.0 - alpha)
            m[M][i] += log(mprob) * ( 1.0 - alpha)
            m[I][i] += log(iprob) * ( 1.0 - alpha)
    res = [[0]*npages, [0]*npages, [0]*npages, [0]*npages]
    dict_ = {C:"C", F:"F", M:"M", I:"I"}
    res[C][0] = m[C][0] #F
    res[F][0] = m[F][0] #F
    res[M][0] = m[M][0] #M
    res[I][0] = m[I][0] #I
    i=0; c="I";
    BT = [[0]*npages, [0]*npages, [0]*npages, [0]*npages]
    for i in range(1, npages):
        #A[j,I]
        #Fanterior + logP(i|j)
        arr = [res[F][i-1] + transProb[("F","I")], res[C][i-1] + transProb[("C","I")]]
        a = np.argmax(arr)
        res[I][i] = arr[a] + m[I][i]
        if a == 0:
            BT[I][i] = F
        else:
            BT[I][i] = C
        #A[j,M]
        arr = [res[I][i-1] + transProb[("I","M")], res[M][i-1] + transProb[("M","M")]]
        a = np.argmax(arr)
        res[M][i] =arr[a] + m[M][i]
        if a == 0:
            BT[M][i] = I
        else:
            BT[M][i] = M
        #A[j,F]
        arr = [res[M][i-1]  + transProb[("M","F")], res[I][i-1]  + transProb[("I","F")]]
        a = np.argmax(arr)
        res[F][i] = arr[a]  + m[F][i]
        if a == 0:
            BT[F][i] = M
        else:
            BT[F][i] = I
        
        #A[j,C]
        #Fanterior + logP(i|j)
        arr = [res[F][i-1] + transProb[("F","C")], res[C][i-1] + transProb[("C","C")]]
        a = np.argmax(arr)
        res[C][i] = arr[a] + m[C][i]
        if a == 0:
            BT[C][i] = F
        else:
            BT[C][i] = C

    
    camino = []
    if open_group:
        a = np.argmax([res[F][-1], res[I][-1], res[M][-1], res[C][-1]])
        if a == 0:
            t = F
        elif a == 1:
            t = I
        elif a == 2:
            t = M
        else:
            t = C
    else:
        if res[F][-1] > res[C][-1]:
            t = F
        else:
            t = C
    errors = 0
    err_msg = ""
    for i in range(len(BT[0])-1, -1, -1):
        err =  dict_[t] != results[i][2]
        errors += err
        if err:
            err_msg = "**"
        else:
            err_msg = ""
        camino.append(dict_[t])
        t = BT[t][i]

    camino = camino[::-1]
    return camino

def calc_prob_transitions(r:list):
    m = {}
    m_prior = {}
    for i in range(len(r)-1):
        t = (r[i], r[i+1])
        m[t] = m.get(t, 0) + 1
        m_prior[r[i]] = m_prior.get(r[i], 0) + 1
    for k,v in m.items():
        t, t2 = k
        m[k] = np.log(v / m_prior[t])
    return m

def cut_segments(r:list, assum_consistency:bool=True):
    if assum_consistency:
        fs = [x for x in r if x == "F"]
        return len(fs)

def get_imgs(path:str):
    all_imgs = []
    for p in ["I", "M", "F", "C"]:
        imgs_path = os.path.join(path, p)
        imgs_list = [x.split("/")[-1].split(".")[0] for x in glob.glob(os.path.join(imgs_path, "*jpg"))]
        all_imgs.extend(imgs_list)
    all_imgs.sort()
    return all_imgs

def raw_process(results:list):
    sequence = []
    for i, (npage, fname, af, ai, am) in enumerate(results):
        sequence.append([fname, rev_dict_.get(np.argmax([af, ai, am]))])
    sequence[0] = [sequence[0][0], "I"]
    sequence[-1] = [sequence[-1][0], "F"]
    return [j for i,j in sequence]

def get_transition_probs(GT_path):
    f = open(GT_path, "r")
    lines = f.readlines()
    f.close()
    sequence = "".join([line.strip().split(" ")[1] for line in lines])
    im_c = sequence.count("IM")
    if_c = sequence.count("IF")
    mm_c = sequence.count("MM")
    mf_c = sequence.count("MF")
    fc_c = sequence.count("FC")
    fi_c = sequence.count("FI")
    cc_c = sequence.count("CC")
    ci_c = sequence.count("CI")
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
    m[("I","F")]=np.log(if_)
    m[("M","M")]=log(mm)
    m[("M","F")]=log(mf)
    m[("F","I")]=np.log(fi)
    m[("F","C")]=np.log(fc)
    m[("C","C")]=np.log(cc)
    m[("C","I")]=np.log(ci)
    return m

def main(p:str, sorted_imgs:list, alg:str, GT:str):
    results, order_gt, res_dict = read_results(p)
    res_text = None
    if alg == "PD":
        m = get_transition_probs(GT)
        res = PD_prob_trans(results, m)
    elif alg == "greedy":
        res = greedy(results)
    elif alg == "raw":
        res = raw_process(results)
    num_exps_pd = cut_segments(res, assum_consistency=True)
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