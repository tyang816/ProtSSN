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

def predict(plm_model, gnn_model, loader):
    gnn_model.eval()
    embed_dict = {}

    with torch.no_grad():
        bar = tqdm(loader)
        for data in bar:
            name = data.protein_name[0]
            bar.set_description(f"Protein: {name}")
            graph_data = plm_model(data)
            esm_embed = graph_data.esm_rep
            out, gnn_embed = gnn_model(graph_data)
            esm_embed, gnn_embed = esm_embed.cpu(), gnn_embed.cpu()
            embed_dict[name] = {"esm_embed": esm_embed, "gnn_embed": gnn_embed}

    return embed_dict


def prepare(args, dataset_name, k, h):
    # for build dataset
    args.mutant_name = f"{dataset_name}_k{k}"
    mutant_dataset = build_mutant_dataset(args)
    protein_names = mutant_dataset.protein_names
    print(f">>> Protein names: {protein_names}")
    mutant_loader = DataLoader(mutant_dataset, batch_size=1, shuffle=False)
    print(f">>> Number of proteins: {len(mutant_dataset)}")
    gnn_model = GNN_model(args)
    print(f">>> k{k}_h{h} {param_num(gnn_model)}")
    gnn_model_path = os.path.join(args.gnn_model_dir, f"protssn_k{k}_h{h}.pt")
    gnn_model.load_state_dict(torch.load(gnn_model_path))
    return args, mutant_loader, gnn_model

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn, or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_model_dir", type=str, default="model/", help="test model name")
    parser.add_argument("--gnn_model_name", type=str, default=None, nargs="+", help="test model name")
    
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    
    # dataset 
    parser.add_argument("--mutant_dataset_dir", type=str, default="data/evaluation", help="mutation dataset")
    parser.add_argument("--mutant_name", type=str, default=None, help="name of mutation dataset")
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
    
    for gnn in args.gnn_model_name:
        k, h = gnn.split("_")
        k, h = int(k[1:]), int(h[1:])
        assert k in [10, 20, 30], f"Invalid k: {k}"
        assert h in [512, 768, 1280], f"Invalid h: {h}"
        print(f"--------------- ProtSSN k{k}_h{h} ---------------")
        args.gnn_config["hidden_channels"] = h
        args.c_alpha_max_neighbors = k
        args, mutant_loader, gnn_model = prepare(args, dataset_name, k, h)
        embed = predict(plm_model=plm_model, gnn_model=gnn_model, loader=mutant_loader)
        torch.save(embed, os.path.join(args.result_dir, f"{gnn}.pt"))