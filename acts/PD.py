import tqdm, numpy as np
from functools import lru_cache

## from: https://gist.github.com/curzona/9435822
# Calculates the levenshtein distance and the edits between two strings
def levenshtein(s1, s2, cost_subs, del_cost, ins_cost, return_rows=False):
  rows = costmatrix(s1, s2, subs_cost=cost_subs, del_cost=del_cost, ins_cost=ins_cost)
  edits = backtrace(s1, s2, rows)
  # for row in rows:
  #   print('\t\t'.join([str(x) for x in row]))
  if not return_rows:
    return rows[-1][-1], edits
  else:
    return rows[-1][-1], edits, rows
 
# Generate the cost matrix for the two strings
# Based on http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def costmatrix2(s1, s2, subs_cost, del_cost, ins_cost):
  rows = []
  
  #Init
  # previous_row = range(len(s2) + 1)
  previous_row = [ins_cost(np.sum(s2[:i+1])) for i in range(len(s2))]
  # previous_row.append(previous_row[-1] + np.sum(s2[-1]))
  previous_row.insert(0, 0)
  print(previous_row)
  # print(previous_row)
  # exit()
  rows.append(list(previous_row))
  # current_row_calculated = [del_cost(np.sum(s1[:i])) for i in range(len(s2))]
  last_del = 0
  for i, c1 in tqdm.tqdm(enumerate(s1)):
    # init2, for deletions
    # current_row = [i + 1]
    d = del_cost(c1)
    current_row = [last_del + d]
    last_del += d
    # current_row = current_row_calculated[:i+1]

    for j, c2 in enumerate(s2):
      ins_cost_j = ins_cost(c2)
      insertions = previous_row[j + 1] + ins_cost_j

      del_cost_j = del_cost(c2)
      deletions = current_row[j] + del_cost_j

      subs_cost_j = subs_cost(c1, c2)
      substitutions = previous_row[j] + subs_cost_j
      
      current_row.append(min(insertions, deletions, substitutions))
      if np.argmin([insertions, deletions, substitutions] == 0):
        print(f"ins {insertions} -> previous_row[j + 1] + ins_cost_j : {previous_row[j + 1]} + {ins_cost_j}")
    previous_row = current_row
 
    rows.append(previous_row)
 
  return rows

def costmatrix(s1, s2, subs_cost, del_cost, ins_cost):
  rows = np.zeros((len(s1)+1,len(s2)+1))
  for i in range(len(s2)):
     rows[0][i+1] = del_cost(s2[:i+1])
  for i in range(len(s1)):
     rows[i+1][0] = ins_cost(s1[:i+1])

  for i, c1 in tqdm.tqdm(enumerate(s1)):
    i += 1
    for j, c2 in enumerate(s2):
      j += 1
      
      ins_cost_j = ins_cost(s1[i-1]) + rows[i-1][j]
      del_cost_j = del_cost(s2[j-1]) +  rows[i][j-1]
      subs_cost_j = subs_cost(s1[i-1], s2[j-1]) + rows[i-1][j-1]
      rows[i,j] = min(ins_cost_j, del_cost_j, subs_cost_j)

  return rows


# Trace back through the cost matrix to generate the list of edits
def backtrace(s1, s2, rows):
  i, j = len(s1), len(s2)
 
  edits = []
 
  while(not (i == 0  and j == 0)):
    prev_cost = rows[i][j]
 
    neighbors = []
 
    if(i!=0 and j!=0):
      neighbors.append(rows[i-1][j-1])
    if(i!=0):
      neighbors.append(rows[i-1][j])
    if(j!=0):
      neighbors.append(rows[i][j-1])
 
    min_cost = min(neighbors)
 
    if(min_cost == prev_cost):
      i, j = i-1, j-1
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


def greedy(results, *kwargs):
    AI, AM, AF = 0,1,2
    dict_ = {
        AI:"I", AF:"F",
        AM:"M"
    }
    camino = []
    npages = len(results)
    camino.append(AI)
    for i, (npage, fname, af, ai, am) in enumerate(results[1:]):
        c = camino[-1]
        if c == AI or c == AM:
            if am > af:
                camino.append(AM)
            else:
                camino.append(AF)
        elif c == AF:
              camino.append(AI)
    if camino[-1] == AI:
        camino[-2] = AM
        camino[-1] = AF
    elif camino[-1] == AM:
        camino[-1] = AF
    for i, c in enumerate(camino):
        camino[i] = dict_[c]
    return camino

def sub_cost(a, b):
    # Following https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4355392 
    # Eq. (6) without norm
    a = np.array(a)
    b = np.array(b)
    diff = np.sum(np.absolute(np.array(a) - np.array(b)))
    Na = np.sum(a)
    Nb = np.sum(b)
    return (np.abs(Na-Nb) + diff) / 2

if __name__ == "__main__":
  a = np.array([
    [1,2],
    [1,1],
    [5,5]
  ])
  b = np.array([
    [6,6],
    [6,6],
    [1,1],
    [1,2],
    # [5,5]
  ])
  # cost = levenshteinDistanceDP(a,b,subs_cost=sub_cost, del_cost=np.sum, ins_cost=np.sum)
  # cost = lev_dist(a,b,subs_cost=sub_cost, del_cost=np.sum, ins_cost=np.sum)
  cost = costmatrix(a,b,subs_cost=sub_cost, del_cost=np.sum, ins_cost=np.sum)
  print(a)
  print(b)
  for i in cost:
    print("\t".join([str(j) for j in i]))
  edits = backtrace(a, b, cost)
  for edit in edits:
    t = edit['type']
    if t == 'substitution':
      print(f"{edit} {a[edit['i']]} -> {b[edit['j']]}")
    elif t == "deletion":
      print(f"{edit} {b[edit['j']]}")
    elif t == "insertion":
      print(f"{edit} {a[edit['i']]}")
    else:
      print("Match")
  
  print(f"Total cost {cost[-1][-1]}")