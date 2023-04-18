### obtaining the results
import argparse, re, os, numpy as np
from get_results import create_acts

def main(args):
    acts_gt, used_pages_gt = create_acts(args.path_gt, used_pages_gt=None)
    lengths_hyp = [len(x) for x in acts_gt]
    print(f"Avg pages per act {np.mean(lengths_hyp)}")
    print(f"Min-max pages per act {np.min(lengths_hyp)}-{np.max(lengths_hyp)}")
    print(f"St-dev pages per act {np.std(lengths_hyp)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the spans')
    parser.add_argument('--path_gt', type=str, help='The span results file')
    args = parser.parse_args()
    main(args)