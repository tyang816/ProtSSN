import argparse
import json
import warnings
import torch
import os
import sys
import yaml
import numpy as np
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader
from numpy import nan
from typing import *
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import logging
from src.models import PLM_model, GNN_model
from src.data import build_mutant_dataset
from src.utils.utils import param_num

# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def label_row(rows, sequence, token_probs, offset_idx=1):
    s = []
    sep = ";"
    if ":" in rows:
        sep = ":"
    for row in rows.split(sep):
        if row.lower() == "wt":
            s.append(0)
            continue
        try:
            wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        except:
            print(f"row: {row}, sequence: {sequence}")
            raise ValueError
        assert sequence[idx] == wt, f"The {row}, {sequence[idx]}"
        wt_encoded, mt_encoded = amino_acids_type.index(wt), amino_acids_type.index(mt)
        score = token_probs[idx, mt_encoded] - token_probs[idx, wt_encoded]
        score = score.item()
        s.append(score)
        
    return sum(s)


def predict(args, plm_model, gnn_model, loader, protein_names):
    gnn_model.eval()
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    spear_cor = np.zeros(protein_num)
    mutation_num = []
    if args.score_name is None:
        args.score_name = f"ProtSSN_k{args.c_alpha_max_neighbors}_h{args.gnn_hidden_dim}" 

    with torch.no_grad():
        for data in loader:
            protein_idx = data.protein_idx
            graph_data = plm_model(data)
            out = gnn_model(graph_data)
            
            seq = "".join([amino_acids_type[i] for i in data.y])
            out = torch.log(softmax(out[:, :20]) + 1e-9)
            mutant_file = os.path.join(args.mutant_dataset_dir, "DATASET", protein_names[data.protein_idx], f"{protein_names[data.protein_idx]}.tsv")
            mutant_df = pd.read_table(mutant_file)
            mutant_df[args.score_name] = mutant_df["mutant"].apply(
                lambda x: label_row(x, seq, out.cpu().numpy())
            )
            result_file = os.path.join(args.result_dir, protein_names[data.protein_idx] + "_labeled.tsv")
            if not os.path.exists(result_file):
                mutant_df.to_csv(result_file, sep="\t", index=False)
                
            result = pd.read_table(result_file)
            result[args.score_name] = mutant_df[args.score_name]
            result.to_csv(result_file, sep="\t", index=False)
            
            mutation_num.append(len(result))
            spear_cor[protein_idx] = spearmanr(
                result["score"], result[args.score_name]
            ).correlation
            print(f"-> {protein_names[protein_idx]}: {spear_cor[protein_idx]}; mutant_num: {len(result)}")
            if spear_cor[protein_idx] is nan:
                spear_cor[protein_idx] = 0
    
    if args.score_info is not None:
        if os.path.exists(args.score_info):
            total_result = pd.read_csv(args.score_info)
            total_result[args.score_name] = spear_cor
            total_result.to_csv(args.score_info, index=False)
        else:
            total_result = {
                "name": protein_names, 
                "count": mutation_num, 
                args.score_name: spear_cor
            }
            total_result = pd.DataFrame(total_result)
            total_result.to_csv(args.score_info, index=False)
    print(f">>> {args.score_name} average spearmanr: {spear_cor.mean()}")

def ensemble(args):
    print("----------------- Ensemble -----------------")
    result_files = os.listdir(args.result_dir)
    for file in tqdm(result_files):
        result_file = os.path.join(args.result_dir, file)
        result_df = pd.read_table(result_file)
        models_pred = [result_df[col].to_list() for col in result_df.columns if col.startswith("ProtSSN")]
        ensemble_pred = np.mean(models_pred, axis=0)
        result_df["ensemble"] = ensemble_pred
        result_df.to_csv(result_file, sep="\t", index=False)
    sp_score = spearmanr(result_df["score"], result_df["ensemble"]).correlation
    print(">>> Ensemble spearmanr: ", sp_score)

def prepare(args, dataset_name):
    args.mutant_name = f"{dataset_name}_k{args.c_alpha_max_neighbors}"
    mutant_dataset = build_mutant_dataset(args)
    protein_names = mutant_dataset.protein_names
    print(f">>> Protein names: {protein_names}")
    mutant_loader = DataLoader(mutant_dataset, batch_size=1, shuffle=False)
    print(f">>> Number of proteins: {len(mutant_dataset)}")
    gnn_model = GNN_model(args)
    print(f">>> k{args.c_alpha_max_neighbors}_h{args.gnn_hidden_dim} {param_num(gnn_model)}")
    gnn_model_path = os.path.join(args.gnn_model_dir, f"protssn_k{args.c_alpha_max_neighbors}_h{args.gnn_hidden_dim}.pt")
    gnn_model.load_state_dict(torch.load(gnn_model_path))
    return args, mutant_loader, protein_names, gnn_model

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn, or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_model_dir", type=str, default="model/", help="test model name")
    parser.add_argument("--gnn_hidden_dim", type=int, default=512, choices=[512, 768, 1280], help="hidden size of gnn")
    
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    parser.add_argument("--use_ensemble", action="store_true", help="use ensemble model")
    
    # dataset 
    parser.add_argument("--mutant_dataset_dir", type=str, default="data/evaluation", help="mutation dataset")
    parser.add_argument("--mutant_name", type=str, default=None, help="name of mutation dataset")
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=10, choices=[10, 20, 30], help="K of dataset")
    
    parser.add_argument("--score_info", type=str, default=None, help="the model output spearmanr score file")
    parser.add_argument("--score_name", type=str, default=None, help="the model output col name")
    parser.add_argument("--result_dir", type=str, default="result/", help="the result output path")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()    
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    
    plm_model = PLM_model(args)
    args.plm_hidden_size = plm_model.model.config.hidden_size
    dataset_name = args.mutant_dataset_dir.split("/")[-1]
    os.makedirs(args.result_dir, exist_ok=True)
        
    if args.use_ensemble:
        for k in [10, 20, 30]:
            for h in [512, 768, 1280]:
                print(f"--------------- ProtSSN k{k}_h{h} ---------------")
                args.gnn_hidden_dim, args.c_alpha_max_neighbors = h, k
                args.gnn_config["hidden_channels"] = args.gnn_hidden_dim
                args, mutant_loader, protein_names, gnn_model = prepare(args, dataset_name)
                predict(
                    args,
                    plm_model=plm_model, gnn_model=gnn_model, 
                    loader=mutant_loader, protein_names=protein_names
                )
        ensemble(args)
    else:
        print(f"--------------- ProtSSN k{args.c_alpha_max_neighbors}_h{args.gnn_hidden_dim} ---------------")
        args.gnn_config["hidden_channels"] = args.gnn_hidden_dim
        args, mutant_loader, protein_names, gnn_model = prepare(args, dataset_name)
        predict(
            args,
            plm_model=plm_model, gnn_model=gnn_model, 
            loader=mutant_loader, protein_names=protein_names
        )