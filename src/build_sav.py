import argparse
import os
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser(description='make single mutant tsv')
parser.add_argument("-d", "--dataset", type=str, default=None)
args = parser.parse_args()

one_letter = {
        'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
        'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
        'GLY':'G', 'PRO':'P', 'CYS':'C'
        }
AA = list(one_letter.values())

base_dir = os.path.join(args.dataset, "DATASET")
proteins = os.listdir(base_dir)
for p in proteins:
    fasta = os.path.join(base_dir, p, f"{p}.fasta")
    seq = open(fasta, "r").readlines()[1].strip()
    data = {"mutant":[], "score":[]}
    for idx,s in enumerate(seq):
        for a in AA:
            if a == s:
                continue
            data["mutant"].append(f"{s}{idx+1}{a}")
            data["score"].append(0)

    print(f"{p} contains { len(data['mutant'])}")
    out_file = os.path.join(base_dir, p, f"{p}.tsv")
    pd.DataFrame(data).to_csv(out_file, sep="\t", index=False)
