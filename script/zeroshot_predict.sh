DATASET=proteingym-benchmark

# use single model for inference (default)
K=10
H=512

CUDA_VISIBLE_DEVICES=0 python zeroshot_predict.py \
    --gnn_model_dir model/ \
    --gnn_hidden_dim $H \
    --c_alpha_max_neighbors $K \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/$DATASET

# use ensemble model for inference
CUDA_VISIBLE_DEVICES=0 python zeroshot_predict.py \
    --gnn_model_dir model/ \
    --use_ensemble \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/$DATASET
