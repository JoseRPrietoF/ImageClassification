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
        fname, _, fprob, iprob, mprob = line.strip().split(" ")
        fprob, iprob, mprob = float(fprob), float(iprob), float(mprob)
        # fprob = max(0.0001, fprob); iprob = max(0.0001, iprob); mprob = max(0.0001, mprob); 
        pnumber = int(fname.split("_")[-1])
        res.append([pnumber, fname, fprob, iprob, mprob])
        hyp_l = rev_dict_.get(np.argmax([fprob, iprob, mprob]))
        res_dict[fname] = [fprob, iprob, mprob]
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

def PD(results):
    npages = len(results)
    F,M,I = 0,1,2
    m = [[0]*npages, [0]*npages, [0]*npages] # F M I 
    for i, (pnumber, fname, gt, fprob, iprob, mprob) in enumerate(results):
        m[F][i] = log(fprob)
        m[M][i] = log(mprob)
        m[I][i] = log(iprob)
    #     print(m[0][i], m[1][i], m[2][i])
    # exit()
    res = [[0]*npages, [0]*npages, [0]*npages]
    dict_ = {F:"F", M:"M", I:"I"}
    res[F][0] = m[F][0] #F
    res[M][0] = m[M][0] #M
    res[I][0] = m[I][0]       #I
    i=0; c="I";
    BT = [[0]*npages, [0]*npages, [0]*npages]
    for i in range(1, npages):
        #A[j,I]
        #Fanterior + logP(i|j)
        res[I][i] = res[F][i-1] + m[I][i]
        BT[I][i] = F
        #A[j,M]
        a = np.argmax([res[I][i-1], res[M][i-1]])
        res[M][i] = [res[I][i-1], res[M][i-1]][a] + m[M][i]
        if a == 0:
            BT[M][i] = I
        else:
            BT[M][i] = M
        #A[j,F]
        a = np.argmax([res[M][i-1], res[I][i-1]])
        res[F][i] = [res[M][i-1], res[I][i-1]][a]  + m[F][i]
        if a == 0:
            BT[F][i] = M
        else:
            BT[F][i] = I

    
    camino = []
    t = F
    errors = 0
    err_msg = ""
    for i in range(len(BT[0])-1, -1, -1):
        err =  dict_[t] != results[i][2]
        errors += err
        if err:
            err_msg = "**"
        else:
            err_msg = ""
        # print(i, dict_[t], results[i][2], results[i][1], err_msg)
        camino.append(dict_[t])
        t = BT[t][i]

    camino = camino[::-1]
    return camino

def PD_prob_trans(results, transProb:dict, res_text:dict=None, alpha=1.0):
    """
    transProb es el dict de probabilidades de transicion
    """
    npages = len(results)
    F,M,I = 0,1,2
    m = [[0]*npages, [0]*npages, [0]*npages] # F M I 
    for i, (pnumber, fname, fprob, iprob, mprob) in enumerate(results):
        m[F][i] = log(fprob) * alpha
        m[M][i] = log(mprob) * alpha
        m[I][i] = log(iprob) * alpha
        if res_text is not None:
            _,fprob,iprob,mprob = res_text[fname]        
            m[F][i] += log(fprob) * ( 1.0 - alpha)
            m[M][i] += log(mprob) * ( 1.0 - alpha)
            m[I][i] += log(iprob) * ( 1.0 - alpha)
    res = [[0]*npages, [0]*npages, [0]*npages]
    dict_ = {F:"F", M:"M", I:"I"}
    res[F][0] = m[F][0] #F
    res[M][0] = m[M][0] #M
    res[I][0] = m[I][0] #I
    i=0; c="I";
    BT = [[0]*npages, [0]*npages, [0]*npages]
    for i in range(1, npages):
        #A[j,I]
        #Fanterior + logP(i|j)
        res[I][i] = res[F][i-1] + m[I][i]
        BT[I][i] = F
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

    
    camino = []
    t = F
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
    for p in ["I", "M", "F"]:
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
    total = im_c + if_c + mm_c + mf_c
    if_ = if_c / (if_c + im_c)
    im = im_c / (if_c + im_c)
    mm = mm_c / (mf_c + mm_c)
    mf = mf_c / (mf_c + mm_c)
    m = {}
    # m[("I","M")]=np.log(0.4949152)
    # m[("I","F")]=np.log(0.5050848)
    # m[("M","M")]=np.log(0.8575610)
    # m[("M","F")]=np.log(0.1424390)
    m[("I","M")]=np.log(im)
    m[("I","F")]=np.log(if_)
    m[("M","M")]=np.log(mm)
    m[("M","F")]=np.log(mf)
    m[("F","I")]=np.log(1)
    # print(f"IF {if_}" )
    # print(f"IM {im}")
    # print(f"MM {mm}")
    # print(f"MF {mf}")
    # print(f"FI {1}")
    # exit()
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
        print(img, "\t ", "\t ", c, f" \t {res_dict[img][0]} \t {res_dict[img][1]} \t {res_dict[img][2]}")

    # print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--model', type=str, help='model')
    parser.add_argument('--folder', type=str, help='folder')
    parser.add_argument('--alg', type=str, help='algorithm')
    parser.add_argument('--GT', type=str, help='algorithm', default="")
    # parser.add_argument('--class_to_cut', type=str, help='Class to use if alg==raw')
    args = parser.parse_args()
    # prod = "4946"
    # model = "resnet18"
    # alg = "greedy"
    alg = args.alg
    prod = args.folder
    model = args.model
    # print(f'Prod for folder {prod}')
    path_res = f"/data2/jose/projects/image_classif/work_JMBD_{prod}_prod_{model}_size1024/results"
    path_imgs = f"/data2/jose/projects/image_classif/data/JMBD4949_4950/prod_{prod}/test"
    
    main(path_res, get_imgs(path_imgs), alg=alg, GT=args.GT)