import argparse
import time
import warnings
import torch
import os, time
import sys
import yaml
import datetime
import logging
import random
import numpy as np
import pandas as pd
import transformers
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from numpy import nan
from typing import *
# from transformers import get_inverse_sqrt_schedule
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator

from src.utils.data_utils import BatchSampler
from src.utils.utils import param_num
from src.models import PLM_model, GNN_model
from src.data import prepare_train_val_dataset

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def create_model(args):
    plm_model = PLM_model(args)
    args.plm_hidden_size = plm_model.model.config.hidden_size
    logger.info("**** Config ****")
    for key, value in vars(args).items():
        print("->",key, value)
    
    gnn_model = GNN_model(args)
    logger.info(param_num(gnn_model))
    
    return plm_model, gnn_model, args


class StepRunner:
    def __init__(self, args, plm_model, gnn_model, 
                 loss_fn, accelerator=None,
                 stage="train", metrics_dict=None,
                 optimizer=None, scheduler=None):
        self.plm_model, self.gnn_model = plm_model, gnn_model
        self.metrics_dict, self.stage = metrics_dict, stage
        self.accelerator = accelerator
        self.optimizer, self.scheduler, self.loss_fn = optimizer, scheduler, loss_fn
        self.args = args
        

    def step(self, batch):
        batch_graph = self.plm_model(batch)
        
        if self.stage == "train":
            with self.accelerator.accumulate(self.gnn_model):
                out = self.gnn_model(batch_graph).cuda()
                y = torch.cat([data.y for data in batch]).to(out.device)
                loss = self.loss_fn(out[:, :20], y)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.gnn_model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()  # Update learning rate schedule
                self.optimizer.zero_grad()
        else:
            out = self.gnn_model(batch_graph).cuda()
            y = torch.cat([data.y for data in batch]).to(out.device)
            loss = self.loss_fn(out[:, :20], y)
        
        # compute metrics
        step_metrics = {}
        if self.metrics_dict and self.stage != 'train':
            # metrics
            step_metrics = {}
            if self.metrics_dict:
                step_metrics = {f"{self.stage}_{name}": metric_fn()
                                for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), self.gnn_model, step_metrics

    def train_step(self, batch):
        self.gnn_model.train()
        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.gnn_model.eval()
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

    def __call__(self, dataloader):
        total_loss = 0
        total_steps = len(dataloader)
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        total_metrics = {}
        epoch_log = {}
        
        for step, batch in loop:
            step_loss, gnn_model, step_metrics = self.steprunner(batch)
            if step == 0 and step_metrics:
                total_metrics = {name: 0 for name in step_metrics.keys()}   
            step_log = dict({f"{self.stage}_loss": round(step_loss, 3)}, **step_metrics)
            
            for name, metric in step_metrics.items():
                total_metrics[name] += metric
            total_loss += step_loss
            
            loop.set_postfix(**step_log)
        
        epoch_loss = total_loss / total_steps
        epoch_metrics = {}
        for name, metric in total_metrics.items():
            epoch_metrics[name] = metric / total_steps
        epoch_log = dict({f"{self.stage}_loss": epoch_loss}, **epoch_metrics)
        loop.set_postfix(**epoch_log)

        return gnn_model, epoch_log



def train_model(args, plm_model, gnn_model, 
                optimizer, scheduler, loss_fn, 
                accelerator=None, metrics_dict=None, 
                train_data=None, valid_data=None,
                monitor="valid_loss", mode="min"):
    history = {"spearman": []}
    start_epoch = 1
    logger.info("***** Running training *****")
    if args.auto_continue_train:
        history_df = pd.read_csv(os.path.join(args.model_dir, "history.csv"))
        names = history_df.columns
        gnn_model.load_state_dict(torch.load(os.path.join(args.model_dir, f"best_{monitor}.pt"))["state_dict"])
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

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(
            args=args, stage="train",
            plm_model=plm_model, gnn_model=gnn_model, 
            loss_fn=loss_fn, accelerator=accelerator,
            metrics_dict=deepcopy(metrics_dict), 
            optimizer=optimizer, scheduler=scheduler
            )
        train_epoch_runner = EpochRunner(train_step_runner)
        gnn_model, train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if valid_data:
            val_step_runner = StepRunner(
                args=args, stage="valid",
                plm_model=plm_model, gnn_model=gnn_model, 
                loss_fn=loss_fn, accelerator=accelerator,
                metrics_dict=deepcopy(metrics_dict), 
                optimizer=optimizer, scheduler=scheduler
                )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                gnn_model, val_metrics = val_epoch_runner(valid_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        
        if best_score_idx == len(arr_scores) - 1:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            model_path = os.path.join(args.model_dir, f"best_{monitor}.pt")
            torch.save({
                "state_dict": gnn_model.state_dict(),
                "epoch": epoch,
                "history": history,
                }, model_path)
            print(f"<<<<<< reach best {monitor} : {'%.3f'%arr_scores[best_score_idx]} >>>>>>")
        
        if args.patience > 0 and len(arr_scores) - best_score_idx > args.patience:
            print(f"<<<<<< {monitor} without improvement in {args.patience} epoch, early stopping >>>>>>")
            break
                
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.model_dir, "history.csv"), index=False)
    


def create_parser():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--noise_ratio", type=float, default=0.05, help="noise probability")
    parser.add_argument("--noise_type", type=str, default="mut", help="mask or mut")
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_hidden_dim", type=int, default=512, help="hidden size of gnn")
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    
    # training strategy
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warm up step")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="warm up percent")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="number of epochs to train")
    parser.add_argument("--epoch_idx", type=int, default=0, help="the idx of epoch to continue training")
    parser.add_argument("--auto_continue_train", action="store_true", help="auto extract epoch idx from history")
    parser.add_argument("--batch_token_num", type=int, default=4096, help="how many tokens in one batch")
    parser.add_argument("--max_graph_token_num", type=int, default=3000, help="max token num a graph has")
    parser.add_argument("--patience", type=int, default=0, help="early stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=4.0, help="clip grad norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    
    # dataset
    parser.add_argument("--cath_dataset", type=str, default="data/cath_k10", help="main protein dataset")
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=10, help="graph dataset K")
    parser.add_argument("--model_dir", type=str, default="", help="which model used to load")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    args.gnn_config["hidden_channels"] = args.gnn_hidden_dim
    args.n_gpu = torch.cuda.device_count()

    # load dataset
    train_dataset, valid_dataset = prepare_train_val_dataset(args)
    def collect_fn(batch):
        return batch
    cath_dataloader = lambda dataset: DataLoader(
        dataset=dataset, num_workers=4, 
        collate_fn=lambda x: collect_fn(x),
        batch_sampler=BatchSampler(
            dataset, 
            max_len=args.max_graph_token_num,
            batch_token_num=args.batch_token_num,
            shuffle=True
            )
        )
    train_dataloader, valid_dataloader = map(
        cath_dataloader, (train_dataset, valid_dataset)
    )
    
    # load model
    plm_model, gnn_model, args = create_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()
    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    optimizer = torch.optim.Adam(
        gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # args.warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)
    # scheduler = get_inverse_sqrt_schedule(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = None
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    gnn_model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        gnn_model, optimizer, train_dataloader, valid_dataloader
    )
    
    os.makedirs(args.model_dir, exist_ok=True)    
    with open(os.path.join(args.model_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False)
    
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_dataset))
    logger.info("  Num valid examples = %d", len(valid_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch token num = %d", args.batch_token_num)
    
    logger.info(
        "  Total train batch token num (w. parallel, distributed & accumulation) = %d",
        args.batch_token_num
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)
    
    # Train!
    set_seed(args)  # Added here for reproducibility
    train_model(
        args=args,
        plm_model=plm_model, gnn_model=gnn_model, 
        optimizer=optimizer, scheduler=scheduler,
        loss_fn=loss_fn, accelerator=accelerator,
        train_data=train_dataloader,
        valid_data=valid_dataloader
        )
    
    # save history
    