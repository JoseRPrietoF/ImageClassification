import os
import numpy as np

def save_to_file(results, tag2label:dict, fname_file:str, fname_file_errors:str, prod:bool=False):
    f = open(fname_file, "w")
    if not prod:
        f_errors = open(fname_file_errors, "w")
    header = "FNAME GT"
    for l,tag in tag2label.items():
        header += f" {tag}"
    f.write(f'{header}\n')
    if not prod:
        f_errors.write(f'{header}\n')

    for fname, output in results:
        c_gt, fname_ = fname.split("/")[-2:]
        if prod:
            c_gt = "-1"
        fname_ = fname_.split(".")[0]
        c_hyp = tag2label[np.argmax(output)]
        output_str = " ".join([str(x) for x in output])
        f.write(f"{fname_} {c_gt} {output_str}\n")
        if c_hyp != c_gt and not prod:
            f_errors.write(f"{fname_} {c_gt} {output_str}\n")
        # print(fname, output)

    f.close()
    if not prod:
        f_errors.close()