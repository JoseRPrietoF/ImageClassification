### obtaining the results
import argparse, re, os, numpy as np
from PD import backtrace
import tqdm

def create_acts(path_file:str, args=None):
    f = open(path_file, "r")
    lines = f.readlines()
    f.close()
    acts = []
    aux = set()
    used_pages = set()
    boundaries = [0]
    for jpage, line in enumerate(lines):
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        num_page = int(fname.split("_")[-1])
        aux.add(fname)

        if c == "F":
            acts.append(aux)
            aux = set()
            boundaries.append(jpage)
    return acts, len(lines), boundaries

def sub_cost(a:set, b:set=set(), weight_cost={}):
    an = a&b
    o = a|b
    r = (o-an ) or (an-o)
    res = 0
    for rpage in r:
       res += weight_cost.get(rpage, 1)
    #    print(weights.keys())
    return res

def sub_cost_seq(bi,bi2, bj, bj2):
    return bi - bi2 + bj - bj2 - 2 * max(0, min(bi, bj) - max(bi2, bj2) )

def costmatrix(s1, s2, subs_cost, del_cost, ins_cost, weight_cost={}):
  rows = np.zeros((len(s1)+1,len(s2)+1))
  for i in range(len(s2)):
     s = set()
     for j in s2[:i+1]:
        s.update(j)
     rows[0][i+1] = del_cost(s, weight_cost=weight_cost)
  for i in range(len(s1)):
     s = set()
     for j in s1[:i+1]:
        s.update(j)
     rows[i+1][0] = ins_cost(s, weight_cost=weight_cost)
  for i, c1 in tqdm.tqdm(enumerate(s1)):
    i += 1
    for j, c2 in enumerate(s2):
      j += 1
      ins_cost_j = ins_cost(s1[i-1], weight_cost=weight_cost) + rows[i-1][j]
      del_cost_j = del_cost(s2[j-1], weight_cost=weight_cost) +  rows[i][j-1]
      
      subs_cost_j = subs_cost(s1[i-1], s2[j-1], weight_cost=weight_cost) + rows[i-1][j-1]
      rows[i,j] = min(ins_cost_j, del_cost_j, subs_cost_j)

  return rows

def costmatrix_seq(s1, s2, subs_cost, del_cost, ins_cost, weight_cost={}):
  rows = np.zeros((len(s1),len(s2)))
  for i in range(1, len(s2)):
     rows[0][i] = rows[0][i-1] + del_cost(s2[i], s2[i-1])
  for i in range(1, len(s1)):
     rows[i][0] = rows[i-1][0] + ins_cost(s1[i], s1[i-1])
  for i, c1 in tqdm.tqdm(enumerate(s1[:-1])):
    i += 1  
    for j, c2 in enumerate(s2[:-1]):
      j += 1
      ins_cost_j = ins_cost(s1[i], s1[i-1]) + rows[i-1][j]
      del_cost_j = del_cost(s2[j], s2[j-1]) +  rows[i][j-1]
      
      subs_cost_j = subs_cost(s1[i], s1[i-1], s2[j], s2[j-1]) + rows[i-1][j-1]
      rows[i,j] = min(ins_cost_j, del_cost_j, subs_cost_j)
  return rows

# Trace back through the cost matrix to generate the list of edits
def backtrace_seq(s1, s2, rows):
  i, j = len(s1)-1, len(s2)-1
 
  edits = []
  # print(rows)
  while(not (i == 0  and j == 0)):
    prev_cost = rows[i][j]
 
    neighbors = []
    # print(edits)
    # print(i,j, prev_cost)
    if(i!=0 and j!=0):
      neighbors.append(rows[i-1][j-1])
    if(i!=0):
      neighbors.append(rows[i-1][j])
    if(j!=0):
      neighbors.append(rows[i][j-1])
 
    min_cost = min(neighbors)
 
    if(min_cost == prev_cost):
      # i, j = i-1, j-1
      i, j = max(0, i-1), max(0, j-1)
      edits.append({'type':'match', 'i':i, 'j':j})
      #   min_cost = 0
    elif(i!=0 and j!=0 and min_cost == rows[i-1][j-1]):
      i, j = i-1, j-1
      edits.append({'type':'substitution', 'i':i, 'j':j})
    elif(i!=0 and min_cost == rows[i-1][j]):
      i, j = i-1, j
      edits.append({'type':'insertion', 'i':i, 'j':j})
    elif(j!=0 and min_cost == rows[i][j-1]):
      i, j = i, j-1
      edits.append({'type':'deletion', 'i':i, 'j':j})
    edits[-1]['cost'] = prev_cost - min_cost
    # print(edits[-1]['cost'], edits[-1]['type'], min_cost)
  edits.reverse()
 
  return edits


def levenshtein(s1, s2, cost_subs, del_cost, ins_cost, return_rows=False, costmatrix=costmatrix, backtrace=backtrace, weight_cost={}):
  rows = costmatrix(s1, s2, subs_cost=cost_subs, del_cost=del_cost, ins_cost=ins_cost, weight_cost=weight_cost)
  edits = backtrace(s1, s2, rows)
  # for row in rows:
  #   print('\t\t'.join([str(x) for x in row]))
  if not return_rows:
    return rows[-1][-1], edits
  else:
    return rows[-1][-1], edits, rows

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

def get_weights(args):
    IG_order = read_IG(args.IG_file, args.num_words)
    f = open(args.path_gt, "r")
    lines = f.readlines()
    f.close()
    acts = []
    res = {}
    total_RW = 0
    for line in lines:
        vector = [0]*(len(IG_order)+1)
        line = line.replace("\t", " ").strip()
        line = re.sub(' +', ' ', line)
        line = line.split(" ")
        fname, c = line[0], line[1]
        page_path_prix = os.path.join(args.path_prix, f"{fname}.idx")
        vector = append_prix(vector, page_path_prix, IG_order)
        vector = np.sum(vector[:-1]) # last component is the "residual" or other words outside of IG
        total_RW += vector
        res[fname] = vector
    for k,v in res.items():
      res[k] = v #/ total_RW
    return res, total_RW

def cost_seq_ins_del(bi, bi2):
   return bi - bi2

def main(args):
   
    acts_gt, npages, boundaries_gt = create_acts(args.path_gt, args=args)
    acts_hyp, _, boundaries_hyp= create_acts(args.path_hyp, args=args)
    weight_cost = {}
    if args.weighted:
       weight_cost, total_RW = get_weights(args)
    if args.calc_seq:
       cost, edits = levenshtein(boundaries_hyp, boundaries_gt, cost_subs=sub_cost_seq, del_cost=cost_seq_ins_del, ins_cost=cost_seq_ins_del, costmatrix=costmatrix_seq, backtrace=backtrace_seq)
    #    print(edits)
       print(f'\n Act Segmentation cost seq : {(cost/npages) * 100.0:.2f}')
    else:
        cost, edits = levenshtein(acts_hyp, acts_gt, cost_subs=sub_cost, del_cost=sub_cost, ins_cost=sub_cost, weight_cost=weight_cost)
        delete_cost, ins_cost, subs_cost = 0, 0, 0
        num_dels, num_ins, num_subs, num_match = 0, 0, 0, 0
        for edit in edits:
            cost = edit['cost']
            if edit['type'] == 'deletion':
                # print(f"delete {edit} i {np.sum(acts_gt[edit['j']])} -> GT {acts_gt[edit['j']]}")
                delete_cost += cost
                num_dels += 1
            elif edit['type'] == 'insertion':
                # print(f"insertion {edit} {np.sum(acts_hyp[edit['i']])} -> HYP {acts_hyp[edit['i']]}")
                ins_cost += cost
                num_ins += 1
            elif edit['type'] == 'substitution':
                # print(f"Subs {edit} GT {acts_gt[edit['j']]} -> HYP {acts_hyp[edit['i']]}")
                subs_cost += cost
                num_subs += 1
            else:
                num_match += 1
        normalization = npages
        if args.weighted:
           normalization = total_RW
        weighted_cost = (delete_cost + ins_cost + subs_cost) / normalization
        if args.weighted:
          print(f'\n Act Segmentation cost weighted : {weighted_cost*100.0:.2f}% ({(delete_cost + ins_cost + subs_cost)} / {normalization}) -> \n [{num_dels}] dels at cost {delete_cost}  \n [{num_ins}] ins at cost {ins_cost} \n [{num_subs}] subs at cost  {subs_cost} \n [{num_match}] matches')
        else:
          print(f'\n Act Segmentation cost : {weighted_cost*100.0:.2f}% ({(delete_cost + ins_cost + subs_cost)} / {normalization}) -> \n [{num_dels}] dels at cost {delete_cost}  \n [{num_ins}] ins at cost {ins_cost} \n [{num_subs}] subs at cost  {subs_cost} \n [{num_match}] matches')

        # print(f'\n Act Segmentation cost : {weighted_cost*100.0:.2f}% ({(delete_cost + ins_cost + subs_cost)} / {normalization}) -> \n [{num_dels}] dels at cost {delete_cost}  \n [{num_ins}] ins at cost {ins_cost} \n [{num_subs}] subs at cost  {subs_cost} \n [{num_match}] matches')
        print(f"GT {len(acts_gt)} = {num_match+num_subs+num_dels} : {len(acts_gt) == num_match+num_subs+num_dels}")
        print(f"HYP {len(acts_hyp)} = {num_match+num_subs+num_ins} : {len(acts_hyp) == num_match+num_subs+num_ins}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--path_hyp', type=str, help='The span results file')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    parser.add_argument('--path_prix', type=str, help='path to save the save')
    parser.add_argument('--IG_file', type=str, help='The span results file')
    parser.add_argument('--num_words', type=int, help='The span results file', default=16384)
    parser.add_argument('--weighted', type=str, help='The span results file', default="False")
    parser.add_argument('--calc_seq', type=str, help='The span results file', default="False")
    args = parser.parse_args()
    args.weighted = args.weighted.lower() in ["true", "si", "t", "yes"]
    args.calc_seq = args.calc_seq.lower() in ["true", "si", "t", "yes"]
    main(args)