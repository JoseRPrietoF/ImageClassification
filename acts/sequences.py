import numpy as np
import copy
from math import log as log_
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
    error = 0
    for line in file.readlines()[1:]:
        # FNAME GT F I M
        fname, gt, fprob, iprob, mprob = line.strip().split(" ")
        fprob, iprob, mprob = float(fprob), float(iprob), float(mprob)
        # fprob = max(0.0001, fprob); iprob = max(0.0001, iprob); mprob = max(0.0001, mprob); 
        pnumber = int(fname.split("_")[-1])
        res.append([pnumber, fname, gt, fprob, iprob, mprob])
        hyp_l = rev_dict_.get(np.argmax([fprob, iprob, mprob]))
        if hyp_l != gt:
            error += 1
    # print(f'{error} errors')
    res.sort()
    file.close()
    order = [i[2] for i in res]
    return res, order

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

def apply_greedy_lm(r):
    results = copy.deepcopy(r)
    res = []
    # for pnumber, fname, gt, fprob, iprob, mprob in results:
    #     res.append(rev_dict_[np.argmax([fprob, iprob, mprob])])
    last = "I"
    res.append("I")
    for pnumber, fname, gt, fprob, iprob, mprob in results[1:]:
        orig = np.argmax([fprob, iprob, mprob])
        if last == "I":
            iprob = 0
        elif last == "F":
            mprob, fprob = 0, 0
        elif last == "M":
            iprob = 0
        new = np.argmax([fprob, iprob, mprob])
        label_new = rev_dict_[new]
        # if orig != new:
        #     print(pnumber)
        res.append(label_new)
        last = label_new
    return res

def apply_greedy_lm2(r):
    results = copy.deepcopy(r)
    res = []
    # for pnumber, fname, gt, fprob, iprob, mprob in results:
    #     res.append(rev_dict_[np.argmax([fprob, iprob, mprob])])
    last = "I"
    res.append("I")
    results = results[1:]
    results[-1][-1] = 0
    results[-1][-2] = 0
    for i, (pnumber, fname, gt, fprob, iprob, mprob) in enumerate(results):
        orig_i = np.argmax([fprob, iprob, mprob])
        lab_i = rev_dict_[orig_i]
        if i+1 < len(results):
            pnumber_2, fname_2, gt_2, fprob_2, iprob_2, mprob_2 = results[i+1]
            orig_2 = np.argmax([fprob_2, iprob_2, mprob_2]) 
            lab_2 = rev_dict_[orig_2]
            if (lab_i, lab_2) not in posibles_states:
                if orig_i > orig_2:
                    label_new = lab_i
                    # results[i+1][3+orig_2] = 0 # aqui
                else:
                    n_arr = [fprob, iprob, mprob]
                    n_arr[orig_i] = 0
                    orig_i = np.argmax(n_arr)
                    label_new = rev_dict_[orig_i]
                    # if (label_new, lab_2) not in posibles_states:
                    #     n_arr[orig_i] = 0
                    #     label_new = rev_dict_[orig_i]
            else:
                label_new = lab_i
        else:
            label_new = lab_i
        res.append(label_new)
        
    return res

def apply_nothing(results):
    res = []
    errors = 0

    for pnumber, fname, gt, fprob, iprob, mprob in results:
        res.append(rev_dict_[np.argmax([fprob, iprob, mprob])])
        # if fprob > iprob and fprob > mprob:
        #     l = "F"
        # elif iprob > fprob and iprob > mprob:
        #     l = "I"
        # else:
        #     l = "M"

        # if res[-1] != gt:
        #     print(fname,res[-1], gt)
        #     errors += 1
        # res.append(l)
        # res.append(gt)

    # print("erors ", errors)
    return res

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
    for i, (pnumber, fname, gt, fprob, iprob, mprob) in enumerate(results):
        m[F][i] = log(fprob) * alpha
        m[M][i] = log(mprob) * alpha
        m[I][i] = log(iprob) * alpha
        if res_text is not None:
            _,fprob,iprob,mprob = res_text[fname]        
            # print(fname, iprob, mprob, fprob)    
            # iprob = np.log(iprob); fprob = np.log(fprob); mprob = np.log(mprob)
            # if alpha == 1:
            #     print("pre", m[F][i])
            # if alpha == 0.9:
            #     print((log(fprob) * ( 1.0 - alpha)), log(fprob), ( 1.0 - alpha), alpha)
            #     exit()
            m[F][i] += log(fprob) * ( 1.0 - alpha)
            m[M][i] += log(mprob) * ( 1.0 - alpha)
            m[I][i] += log(iprob) * ( 1.0 - alpha)
            # if alpha == 1:
            #     print("post", m[F][i], ( 1.0 - alpha))
    #     print(m[0][i], m[1][i], m[2][i])
    # exit()
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
        # print(i, dict_[t], results[i][2], results[i][1], err_msg)
        camino.append(dict_[t])
        t = BT[t][i]

    camino = camino[::-1]
    
    # c = camino[0]
    # for c2 in camino[1:]:
    #     if (c, c2) not in posibles_states:
    #         print(c,c2)
    #     c = c2

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

def main(p:str, p_te:str, path_res_tr_text:str=None):
    results, order_gt = read_results(p)
    _, order_gt_tr = read_results(p_te)
    res_text = None; res_arr_text=None
    if path_res_tr_text is not None:
        res_arr_text, res_text = read_results_text(path_res_tr_text)
    # m = {}
    # m[("I","M")]=np.log(0.4949152)
    # m[("I","F")]=np.log(0.5050848)
    # m[("M","M")]=np.log(0.8575610)
    # m[("M","F")]=np.log(0.1424390)
    m = calc_prob_transitions(order_gt_tr)
    res_greedy = apply_greedy_lm(results)
    res_greedy2 = apply_greedy_lm2(results)
    res_orig = apply_nothing(results)
    res_pd = PD(results)
    res_pd_probs = PD_prob_trans(results, m)
    res_pd_onlyText = PD(res_arr_text)
    res_pd_probs_onlyText = PD_prob_trans(res_arr_text, m)
    # print(res_pd_probs_onlyText)
    
    num_exps_gt = cut_segments(order_gt, assum_consistency=True)
    num_exps_pd = cut_segments(res_pd, assum_consistency=True)
    num_exps_pd_onlyText = cut_segments(res_pd_onlyText, assum_consistency=True)
    num_exps_pd_probs = cut_segments(res_pd_probs, assum_consistency=True)
    
    num_exps_pd_probs_onlyText = cut_segments(res_pd_probs_onlyText, assum_consistency=True)
    # errors = 0
    # for i,j in zip(order_gt, res_orig):
    #     errors += i != j
    # print(f'{errors} / {len(order_gt)} ({errors / len(order_gt)} % of classif error without LM)')
    # errors = 0
    # for i,j in zip(order_gt, res_greedy):
    #     errors += (i != j)
    # print(f'{errors} / {len(order_gt)} ({errors / len(order_gt)} % of classif error)')
    # errors = 0
    # for i,j in zip(order_gt, res_greedy2):
    #     errors += i != j
    #     # print(i, j)
    # print(f'{errors} / {len(order_gt)} ({errors / len(order_gt)} % of classif error greedy2)')
    print(f'{num_exps_gt} expedients in GT')
    errors = 0
    for r,j in zip(results, res_pd):
        i = r[2]
        errors += i != j
        # if i != j:
        #     print(r, i, j)
    print(f'{errors} / {len(order_gt)} ({(errors / len(order_gt))*100} % of classif error prog dyn) - {num_exps_pd} expedients -> {num_exps_pd - num_exps_gt} NumSegmErr ')


    errors = 0
    for r,j in zip(results, res_pd_probs):
        i = r[2]
        errors += i != j
        # if i != j:
        #     print(r, i, j)
    print(f'{errors} / {len(order_gt)} ({(errors / len(order_gt))*100} % of classif error prog dyn with probs of transitions precalculated) -  {num_exps_pd_probs} expedients -> {num_exps_pd_probs - num_exps_gt} NumSegmErr ')

    errors = 0
    for r,j in zip(results, res_pd_onlyText):
        i = r[2]
        errors += i != j
        # if i != j:
        #     print(r, i, j)
    print(f'{errors} / {len(order_gt)} ({(errors / len(order_gt))*100} % of classif error prog dyn with probs from ONLY TEXT) -  {num_exps_pd_onlyText} expedients -> {num_exps_pd_onlyText - num_exps_gt} NumSegmErr ')
    # print(res_pd)

    errors = 0
    for r,j in zip(results, res_pd_probs_onlyText):
        i = r[2]
        errors += i != j
        # if i != j:
        #     print(r, i, j)
    print(f'{errors} / {len(order_gt)} ({(errors / len(order_gt))*100} % of classif error prog dyn with probs of transitions precalculated and probs from ONLY text classifier) -  {num_exps_pd_probs_onlyText} expedients -> {num_exps_pd_probs_onlyText - num_exps_gt} NumSegmErr ')
    # exit()
    print("-----------------------------")
    print("alp  #leg err #pag  err(%)")
    print("-----------------------------")
    for alpha in range(0,101,10):
        # alpha = 1
        alpha /= 100
        res_pd_probs_text = PD_prob_trans(results, m, res_text, alpha=alpha)
        num_exps_pd_probs_text = cut_segments(res_pd_probs_text, assum_consistency=True)
        errors = 0
        for r,j in zip(results, res_pd_probs_text):
            i = r[2]
            errors += i != j
            # if i != j:
            #     print(r, i, j)
        print(f"{alpha}  {num_exps_pd_probs_text - num_exps_gt}  {errors}   {len(order_gt)}  {(errors / len(order_gt))*100:2f}")
        # print(f'ALPHA {alpha} {errors} / {len(order_gt)} ({(errors / len(order_gt))*100} % of classif error prog dyn with probs of transitions precalculated and probs from text classifier) -  {num_exps_pd_probs_text} expedients -> {num_exps_pd_probs_text - num_exps_gt} NumSegmErr ')

    print("-----------------------------")

    # for num, (j, i, r) in enumerate(zip(res_greedy2, order_gt, results)):
    #     if i == j:
    #         print(r[1], i, j, r[2:])
    #     else:
    #         print(r[1], i, j, "*", r[2:])


if __name__ == "__main__":
    te = "50"; tr = "49"
    # te = "49"; tr = "50"
    print(f'Training {tr} and testing with {te}')
    path_res_te = f"/data2/jose/projects/image_classif/work_JMBD4949_4950_tr{te}_resnet50/results"
    path_res_tr = f"/data2/jose/projects/image_classif/work_JMBD4949_4950_tr{tr}_resnet50/results"
    path_res_tr_text = f'/data2/jose/projects/docClasifIbPRIA22/works_IMF/work_tr49_128,128_numFeat1024/results_IMF.txt'
    main(path_res_tr, path_res_te, path_res_tr_text=path_res_tr_text)