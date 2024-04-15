
path_hyp = "/data2/jose/projects/myserver/data/FG_1574"
path_corrections = "/data2/jose/projects/myserver/data/FG_1574_modified"

def read_files(path:str):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    res_dict, res_ord_list = {}, []
    for line in lines:
        fname, c = line.strip().split(" ")
        res_ord_list.append([fname, c])
        res_dict[fname] = c
    return res_dict, res_ord_list

if __name__ == "__main__":
    _, res_ord_list = read_files(path_hyp)
    res_dict, _ = read_files(path_corrections)
    for i, (fname, c) in enumerate(res_ord_list):
        if fname in res_dict:
            # print(fname, c, res_dict[fname])
            res_ord_list[i] = [fname, res_dict[fname]]
    for i, (fname, c) in enumerate(res_ord_list):
        print(fname, c)