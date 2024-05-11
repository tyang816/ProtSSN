import argparse
import time
import warnings
import torch
import os, time
import sys
import yaml
import wandb
import datetime
import logging
import random
import numpy as np
import pandas as pd
import transformers
import json
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import *
# from transformers import get_inverse_sqrt_schedule
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from accelerate.utils import set_seed
from accelerate import Accelerator
from torchmetrics.classification import Accuracy
from src.models import ProtssnClassification, PLM_model, GNN_model
from src.utils.data_utils import BatchSampler
from src.utils.utils import param_num, total_param_num
from src.dataset.supervise_dataset import SuperviseDataset
from src.utils.dataset_utils import NormalizeProtein

# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
# ignore warning information
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 3 + "%s" % nowtime + "==========" * 3)
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, args, model, 
                 loss_fn, accelerator=None,
                 stage="train", metrics_dict=None,
                 optimizer=None, scheduler=None):
        self.model = model
        self.metrics_dict, self.stage = metrics_dict, stage
        self.accelerator = accelerator
        self.optimizer, self.scheduler, self.loss_fn = optimizer, scheduler, loss_fn
        self.args = args

    def step(self, batch):        
        if self.stage == "train":
            with self.accelerator.accumulate(self.model):
                logits = self.model(batch).cuda()
                label = torch.cat([data.label for data in batch]).to(logits.device)
                loss = self.loss_fn(logits, label)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.args.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.pooling_head.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()  # Update learning rate schedule
                self.optimizer.zero_grad()
        else:
            logits = self.model(batch).cuda()
            label = torch.cat([data.label for data in batch]).to(logits.device)
            loss = self.loss_fn(logits, label)
        
        # compute metrics
        if self.metrics_dict and self.stage != "train":
            for name, metric_fn in self.metrics_dict.items():
                metric_fn.update(logits, label)
        return loss.item(), self.model, self.metrics_dict

    def train_step(self, batch):
        self.model.train()
        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        return self.step(batch)

    def __call__(self, batch):
        if self.stage == "train":
            return self.train_step(batch)
        else:
            return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.args = steprunner.args

    def __call__(self, dataloader):
        loop = tqdm(dataloader, total=len(dataloader), file=sys.stdout)
        total_loss = 0
        for batch in loop:
            step_loss, model, metrics_dict = self.steprunner(batch)
            step_log = dict({f"{self.stage}/loss": round(step_loss, 3)})
            if self.args.wandb and self.stage == "train":
                wandb.log({f"train/loss": step_loss, "train/epoch": self.args.epoch_idx})
            loop.set_postfix(**step_log)
            total_loss += step_loss
        
        for name, metric_fn in metrics_dict.items():
            epoch_metric_results = {f"{self.stage}/{name}": metric_fn.compute().item()}
            metric_fn.reset()
        avg_loss = total_loss / len(dataloader)
        epoch_metric_results[f"{self.stage}/epoch_loss"] = avg_loss
        return model, epoch_metric_results

def train_model(args, model, 
                optimizer, scheduler, loss_fn, 
                accelerator=None, metrics_dict=None, 
                train_data=None, valid_data=None, test_data=None,
                monitor="valid/loss", mode="min"):
    history = {}
    start_epoch = 1
    model_path = os.path.join(args.model_dir, args.model_name)
    logger.info("***** Running training *****")
    if args.auto_continue_train:
        history_df = pd.read_csv(os.path.join(args.model_dir, "history.csv"))
        names = history_df.columns
        model.pooling_head.load_state_dict(torch.load(model_path)["state_dict"])
        if args.epoch_idx:
            logger.info(f" Train from epoch_idx = {args.epoch_idx} ")
        else:
            if mode == "min":
                args.epoch_idx = int(history_df[history_df[monitor] == history_df[monitor].min()]["epoch"])
            elif mode == "max":
                args.epoch_idx = int(history_df[history_df[monitor] == history_df[monitor].max()]["epoch"])
            logger.info(f"  Auto continue to train from epoch_idx = {args.epoch_idx} ")
        for name in names:
            history[name] = list(history_df[name][:int(args.epoch_idx)])       
        start_epoch += args.epoch_idx
        
    for epoch in range(start_epoch, args.num_train_epochs + 1):
        printlog(f"Epoch {epoch} / {args.num_train_epochs}")
        args.epoch_idx = epoch
        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(
            args=args, stage="train", model=model, 
            loss_fn=loss_fn, accelerator=accelerator,
            metrics_dict=deepcopy(metrics_dict), 
            optimizer=optimizer, scheduler=scheduler
            )
        train_epoch_runner = EpochRunner(train_step_runner)
        model, epoch_metric_results = train_epoch_runner(train_data)

        for name, metric in epoch_metric_results.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if valid_data:
            val_step_runner = StepRunner(
                args=args, stage="valid", model=model, 
                loss_fn=loss_fn, accelerator=accelerator,
                metrics_dict=deepcopy(metrics_dict), 
                optimizer=optimizer, scheduler=scheduler
                )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                model, epoch_metric_results = val_epoch_runner(valid_data)
            
            if args.wandb:
                wandb.log({name: metric for name, metric in epoch_metric_results.items()})
            for name, metric in epoch_metric_results.items():
                print(f">>> Epoch {epoch} {name}: {'%.3f'%metric}")
            
            epoch_metric_results["epoch"] = epoch
            for name, metric in epoch_metric_results.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save({
                "state_dict": model.pooling_head.state_dict(),
                "epoch": epoch,
                "history": history,
                }, model_path)
            print(f">>> reach best {monitor} : {'%.3f'%arr_scores[best_score_idx]}")
        
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.model_dir, "history.csv"), index=False)
        
        if args.patience > 0 and len(arr_scores) - best_score_idx > args.patience:
            print(f">>> {monitor} without improvement in {args.patience} epoch, early stopping")
            break
        
        # 4，test -------------------------------------------------
        if test_data:
            test_step_runner = StepRunner(
                args=args, stage="test", model=model, 
                loss_fn=loss_fn, accelerator=accelerator,
                metrics_dict=deepcopy(metrics_dict), 
                optimizer=optimizer, scheduler=scheduler
                )
            test_epoch_runner = EpochRunner(test_step_runner)
            with torch.no_grad():
                model, epoch_metric_results = test_epoch_runner(test_data)
            for name, metric in epoch_metric_results.items():
                print(f">>> Epoch {epoch} {name}: {'%.3f'%metric}")
            if args.wandb:
                wandb.log({name: metric for name, metric in epoch_metric_results.items()})


def create_parser():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_hidden_dim", type=int, default=512, help="hidden size of gnn")
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    parser.add_argument("--plm_hidden_size", type=int, default=1280, help="hidden size of plm")
    parser.add_argument("--pooling_method", type=str, default="mean", help="pooling method")
    parser.add_argument("--pooling_dropout", type=float, default=0.1, help="pooling dropout")
    
    # training strategy
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="number of epochs to train")
    parser.add_argument("--epoch_idx", type=int, default=0, help="the idx of epoch to continue training")
    parser.add_argument("--auto_continue_train", action="store_true", help="auto extract epoch idx from history")
    parser.add_argument("--batch_token_num", type=int, default=4096, help="how many tokens in one batch")
    parser.add_argument("--max_graph_token_num", type=int, default=3000, help="max token num a graph has")
    parser.add_argument("--patience", type=int, default=0, help="early stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="clip grad norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    
    # dataset
    parser.add_argument("--num_labels", type=int, help="number of labels")
    parser.add_argument("--problem_type", type=str, default="classification", help="classification or regression")
    parser.add_argument("--supv_dataset", type=str, help="supervise protein dataset")
    parser.add_argument("--train_file", type=str, help="train label file")
    parser.add_argument("--valid_file", type=str, help="valid label file")
    parser.add_argument("--test_file", type=str, help="test label file")
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=10, help="graph dataset K")
    parser.add_argument("--gnn_model_path", type=str, default="", help="gnn model path")
    
    # save model
    parser.add_argument("--model_dir", type=str, default="model", help="model save dir")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    
    # log
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="protssn", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    args.gnn_config["hidden_channels"] = args.gnn_hidden_dim
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"ProtSSN-task"
        if args.model_name is None:
            args.model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # load dataset
    logger.info("***** Loading Dataset *****")
    datatset_name = args.supv_dataset.split("/")[-1]
    pdb_dir = f"{args.supv_dataset}/esmfold_pdb"
    graph_dir = f"{datatset_name}_k{args.c_alpha_max_neighbors}"
    supervise_dataset = SuperviseDataset(
        root=args.supv_dataset,
        raw_dir=pdb_dir,
        name=graph_dir,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pre_transform=NormalizeProtein(
            filename=f'norm/cath_k{args.c_alpha_max_neighbors}_mean_attr.pt'
        ),
    )

    label_dict = {}
    def get_dataset(df):
        names, node_nums = [], []
        for name, label, seq in zip(df["name"], df["label"], df["sequence"]):
            names.append(name)
            label_dict[name] = label
            node_nums.append(len(seq))
        return names, node_nums
    train_names, train_node_nums = get_dataset(pd.read_csv(args.train_file))
    valid_names, valid_node_nums = get_dataset(pd.read_csv(args.valid_file))
    test_names, test_node_nums = get_dataset(pd.read_csv(args.test_file))
    
    
    def process_data(name):
        data = torch.load(f"{args.supv_dataset}/{graph_dir.capitalize()}/processed/{name}.pt")
        data.label = torch.tensor(label_dict[name]).view(1)
        return data
    
    def collect_fn(batch):
        batch_data = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_data, name) for name in batch]
            for future in as_completed(futures):
                graph = future.result()
                batch_data.append(graph)
        return batch_data
    
    train_dataloader = DataLoader(
        dataset=train_names, num_workers=4, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            node_num=train_node_nums,
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=True
            )
        )
    valid_dataloader = DataLoader(
        dataset=valid_names, num_workers=4, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            node_num=valid_node_nums,
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=False
            )
        )
    test_dataloader = DataLoader(
        dataset=test_names, num_workers=4, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            node_num=test_node_nums,
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=False
            )
        )
    
    logger.info("***** Load Model *****")
    # load model
    plm_model = PLM_model(args)
    gnn_model = GNN_model(args)
    gnn_model.load_state_dict(torch.load(args.gnn_model_path))
    protssn_classification = ProtssnClassification(args, plm_model, gnn_model)
    protssn_classification.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for param in plm_model.parameters():
        param.requires_grad = False
    for param in gnn_model.parameters():
        param.requires_grad = False
    logger.info(total_param_num(protssn_classification))
    logger.info(param_num(protssn_classification))
    optimizer = torch.optim.AdamW(
        protssn_classification.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = None
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    protssn_classification, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        protssn_classification, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    metrics_dict = {
        "acc": Accuracy(task="multiclass", num_classes=args.num_labels).to(device)
    }
    
    os.makedirs(args.model_dir, exist_ok=True)    
    with open(os.path.join(args.model_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False)
    
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_names))
    logger.info("  Num valid examples = %d", len(valid_names))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch token num = %d", args.batch_token_num)
    
    logger.info(
        "  Total train batch token num (w. parallel, distributed & accumulation) = %d",
        args.batch_token_num
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    
    train_model(
        args=args, model=protssn_classification, 
        optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, 
        accelerator=accelerator, metrics_dict=metrics_dict, 
        train_data=train_dataloader, valid_data=valid_dataloader, test_data=test_dataloader,
        monitor="valid/acc", mode="max"
        )
    if args.wandb:
        wandb.finish()    