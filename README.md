# Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability

- [Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability](#semantical-and-topological-protein-encoding-toward-enhanced-bioactivity-and-thermostability)
  - [🚀 Introduction (ProtSSN)](#-introduction-protssn)
  - [📑 Results](#-results)
    - [News](#news)
    - [Downloads](#downloads)
    - [Paper Results](#paper-results)
  - [🛫 Requirement](#-requirement)
    - [Conda Enviroment](#conda-enviroment)
    - [Hardware](#hardware)
  - [🧬 Zero-shot Prediction for Mutants](#-zero-shot-prediction-for-mutants)
    - [Download Pre-trained Checkpoints](#download-pre-trained-checkpoints)
    - [Prepare Your Own Dataset](#prepare-your-own-dataset)
    - [Test on Your Own Dataset](#test-on-your-own-dataset)
    - [Directed Evolution for Proteins Without Experimental Data](#directed-evolution-for-proteins-without-experimental-data)
  - [🤖 Fine-tuning on Downstream Tasks](#-fine-tuning-on-downstream-tasks)
    - [How to Fine-tune ProtSSN](#how-to-fine-tune-protssn)
    - [Find More Datasets](#find-more-datasets)
  - [🔬 Get Embeddings for Analysis](#-get-embeddings-for-analysis)
    - [Prepare Your Own Dataset](#prepare-your-own-dataset-1)
    - [Start Extracting Embeddings](#start-extracting-embeddings)
  - [✏️ Pre-train From Scratch](#️-pre-train-from-scratch)
    - [Download Pre-train Dataset](#download-pre-train-dataset)
    - [Build CATH Graph (optional)](#build-cath-graph-optional)
    - [Start Training](#start-training)
  - [🙌 Citation](#-citation)

## 🚀 Introduction (ProtSSN)

Fusion of protein **sequence and structural** information, using **denoising** pre-training network for **protein engineering** (**zero-shot**). Please note that we do not require MSA or evolutionary information.

We focus on using end-to-end methods for protein directed evolution in zero sample scenarios, our overall framework can be shown in the following figure.

<img src="img/architecture.png" alt="Logo">

## 📑 Results

### News

- [2024.4.28] Our paper is under review on eLife, 10.7554/eLife.98033.
- [2023.12.23] Our **ensemble ProtSSN** achieves a spearman score of **0.449** on **ProteinGym v1.0**. You can view and compare different baseline models on the [ProteinGym website](https://proteingym.org/benchmarks).

### Downloads

- ProteinGym: The pdb files folded by ColabFold 1.5 can be downloaded from https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip
- ProteinGym: The pdb and mutant files in ProtSSN formation can be downloaded from https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_ProtSSN.zip
- **DTM** and **DDG** dataset can be found in [`data/DTM`](https://github.com/tyang816/ProtSSN/tree/master/data/DTM) and [`data/DDG`](https://github.com/tyang816/ProtSSN/tree/master/data/DDG)

### Paper Results

We conducted tests on **ProteinGym** and compared it with the non-MSA method. Meanwhile, due to the fusion of structural information, we also conducted tests on two newly constructed stability datasets, **DTm** and **DDG**, in order to verify the correlation between structure and stability.

<img src="img/tab3.png" alt="tab3">

Our model surpasses most MSA methods and almost all language model methods **without considering viruses**, which means you can confidently use our model end-to-end. At the same time, because there is **no need for MSA** for reasoning or training, this **reduces the tedious process and prior knowledge** of collecting MSA, and there is **no potential performance instability caused by MSA** search methods/quantities.

<img src="img/tab5.png" alt="tab5">

## 🛫 Requirement

### Conda Enviroment

Please make sure you have installed **[Anaconda3](https://www.anaconda.com/download)** or **[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)**.

**Enviroment.**

```shell
conda env create -f environment.yaml
conda activate protssn
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### Hardware

- For direct use of **inference**, we recommend at least **10G** of graphics memory, such as RTX 3080
- For **retraining** a ProtSSN, we recommend that the larger the graphics memory. We recommend a **24GB** RTX 3090 or better.

## 🧬 Zero-shot Prediction for Mutants

### Download Pre-trained Checkpoints

We have prepared 9 models that you can use separately or integrate. 😉 

If you only want to only use **one model** for reasoning, we recommend using **k20_h512**.

| # Version | # Param | # Link                                                       |
| --------- | ------- | ------------------------------------------------------------ |
| k10_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h512.pt |
| k10_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h768.pt |
| k10_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k10_h1280.pt |
| k20_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h512.pt |
| k20_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h768.pt |
| k20_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h1280.pt |
| k30_h512  | 148     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h512.pt |
| k30_h768  | 160     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h768.pt |
| k30_h1280 | 184     | https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k30_h1280.pt |

```shell
mkdir model
cd model
wget https://huggingface.co/tyang816/ProtSSN/resolve/main/protssn_k20_h512.pt
```

Ensembling models can achieve better results, and given our lower reasoning costs, you can use **ensemble methods** for zero-shot prediction. 😊

`ProtSSN.zip` contains all the model checkpoints, our **training records** and **configs** can be found in `model/history` and `model/config`.

```shell
wget https://huggingface.co/tyang816/ProtSSN/resolve/main/ProtSSN.zip
unzip ProtSSN.zip
rm ProtSSN.zip
```

### Prepare Your Own Dataset

Here is a subset of ProtetinGym as the basic sample for mutation prediction in [`data/mutant_example`](https://github.com/tyang816/ProtSSN/tree/master/data/mutant_example), the files should be arranged in the following format:

```
data/proteingym-benchmark
|——DATASET
|——|——Protein1
|——|——|——Protein1.pdb
|——|——|——Protein1.tsv
|——|——Protein2
|——|——...
```

- Because our model requires **structure (PDB)** as input, we recommend using [**Alphafold**](https://github.com/google-deepmind/alphafold) for folding, and of course, **[ESMFold](https://github.com/facebookresearch/esm)** is also a good choice.
- If you do not have a tool installed for folding protein structures, you can search for your protein by Uniprot ID in the **AlphaFold database** (https://alphafold.ebi.ac.uk/) without consuming resources for folding.

### Test on Your Own Dataset

The following script can be found at `script/zeroshot_predict.sh`, more details can be found at `zeroshot_predict.py`. We recommend using ensemble method for inference, which can be completed with a single RTX3080.

```shell
# k in (10, 20, 30)
# h in (512, 768, 1280)
# use single model for inference (default)
DATASET=proteingym-benchmark
CUDA_VISIBLE_DEVICES=0 python zeroshot_predict.py \
    --gnn_model_name k10_h512 \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/$DATASET


# select the models for ensemble prediction
DATASET=proteingym-benchmark
CUDA_VISIBLE_DEVICES=0 python zeroshot_predict.py \
    --gnn_model_name k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280 \
    --use_ensemble \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/$DATASET
```

### Directed Evolution for Proteins Without Experimental Data

In this case, we suggest that you start from a **single point** position, and you still need to organize your protein sequence and structural files in the same way as above.

You can use the following script to construct data for **single point saturation mutations** and use models for prediction.

```shell
python src/build_sav.py -d data/mutant_example/no_exp
# output: A0A5J4FAY1_MICAE contains 14193
```

If you want to use ProtSSN for higher-level mutations, we recommend that you use a method that considers both rational design and AI design.

At the same time, the mutant column of the CSV or TSV data should be split by ":", for example, "M1V:A2V".

## 🤖 Fine-tuning on Downstream Tasks

### How to Fine-tune ProtSSN

We provide how to fine-tune ProtSSN models on downstream tasks.

1. You could prepare your own dataset formation in [`data/finetune_example`](https://github.com/tyang816/ProtSSN/tree/master/data/finetune_example), the CSV file which contains **label** and corresponding **pdb files** are essential.
2. You can follow the [`script/run_ft.sh`](https://github.com/tyang816/ProtSSN/blob/master/script/run_ft.sh) to fine-tune the ProtSSN model, we recommend the setting of **k20_512**.

### Find More Datasets

For more downstream task dataset we recommend to use the datasets from `SES-Adapter` which we collect lots of dataset, such as EC, GO, MetalIonBinding, ProtSolM and other datasets with their pdb files, .

1. PDB files can be downloaded at https://huggingface.co/datasets/tyang816/ProtFactory-PDB/tree/main.
2. Please see my [huggingface site](https://huggingface.co/tyang816) to get task CSV files.

If you use the additional datasets we prepare, please cite the SES-Adapter, thanks!

```shell
@article{tan2024ses-adapter,
  title={Simple, efficient and scalable structure-aware adapter boosts protein language models},
  author={Tan, Yang and Li, Mingchen and Zhou, Bingxin and Zhong, Bozitao and Zheng, Lirong and Tan, Pan and Zhou, Ziyi and Yu, Huiqun and Fan, Guisheng and Hong, Liang},
  journal={arXiv preprint arXiv:2404.14850},
  year={2024}
}
```

## 🔬 Get Embeddings for Analysis

Perhaps you need protein embeddings for evolutionary or other direction analysis.

### Prepare Your Own Dataset

Here is a basic sample for embedding extraction in [`data/mutant_example`](https://github.com/tyang816/ProtSSN/tree/master/data/mutant_example), the files should be arranged in the following format:

```
data/proteingym-benchmark
|——DATASET
|——|——Protein1
|——|——|——Protein1.pdb
|——|——|——Protein1.tsv (no need)
|——|——Protein2
|——|——...
```

### Start Extracting Embeddings

```shell
DATASET=proteingym-benchmark
CUDA_VISIBLE_DEVICES=0 python get_embedding.py \
    --gnn_model_name k10_h512 k20_h512 \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/embed/$DATASET
```

The output file is `<model>.pt` which is the following format: 

*esm_embed* is the embedding from facebook/esm2_650M, *protssn_embed* is the embedding from protssn which fusues the protein language and sturcture information.

```shell
{
    protein1: {"esm_embed": esm_embed, "protssn_embed": protssn_embed}, 
    protein2: {"esm_embed": esm_embed, "protssn_embed": protssn_embed},
    ...
}
```

## ✏️ Pre-train From Scratch

### Download Pre-train Dataset

We use `CATH` for pre-training, you can find more details in https://www.cathdb.info/.

```shell
mkdir -p data/cath
cd data/cath
wget https://huggingface.co/datasets/tyang816/cath/blob/main/dompdb.tar
# or wget https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/dompdb.tar
tar -xvf dompdb.tar
rm dompdb.tar
```

### Build CATH Graph (optional)

You can first create CATH graphs, but if you skip this step, the graphs will be automatically created in start training step.

see `script/build_cath_dataset.sh`

```shell
# k can be any number if you want
K=30

python src/data.py \
    --data_type cath \
    --c_alpha_max_neighbors $K \
    --cath_dataset data/cath/cath_k$K
```

### Start Training

You can view our work as an **adapter** to enhance the capabilities of language models, so you can replace any language model as a prefix.

see `script/run_pt.sh`

```shell
K=20
H=512

CUDA_VISIBLE_DEVICES=0 \
python run_pt.py \
    --seed 12345 \
    --noise_ratio 0.05 \
    --noise_type mut \
    --gnn egnn \
    --gnn_config src/config/egnn.yaml \
    --gnn_hidden_dim $H \
    --plm facebook/esm2_t33_650M_UR50D \
    --cath_dataset data/cath/cath_k$K \
    --c_alpha_max_neighbors $K \
    --learning_rate 1e-4 \
    --warmup_percent 0 \
    --weight_decay 1e-4 \
    --num_train_epochs 100 \
    --batch_token_num 1024 \
    --max_grad_norm 4 \
    --gradient_accumulation_steps 10 \
    --patience 20 \
    --model_dir debug/model/"k"$K"_h"$H
```

## 🙌 Citation

Please cite our work if you have used our code or data for **dry experiment testing/wet experiment directed evolution**. We are pleased to see improvements in the subsequent work.

```
@article{tan2023protssn,
  title={Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability},
  author={Tan, Yang and Zhou, Bingxin and Zheng, Lirong and Fan, Guisheng and Hong, Liang},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

