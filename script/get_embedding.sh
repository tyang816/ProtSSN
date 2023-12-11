DATASET=proteingym-benchmark
CUDA_VISIBLE_DEVICES=0 python get_embedding.py \
    --gnn_model_name k10_h512 k20_h512 \
    --mutant_dataset_dir data/mutant_example/$DATASET \
    --result_dir result/embed/$DATASET