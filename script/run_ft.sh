K=20
H=512
pooling_method=attention1d
# your fine-tuning dataset
dataset_dir=data/finetune_example/PDBSol
pdb_dir_name=esmfold_pdb
# where to save your fine-tuned model
output_model_dir=result/PDBSol/protssn_k"$K"_h"$H"
output_model_name=protssn_"$pooling_method".pt
CUDA_VISIBLE_DEVICES=0 python run_ft.py \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --pooling_method $pooling_method \
    --output_model_dir $output_model_dir \
    --output_model_name $output_model_name \
    --num_labels 2 \
    --supv_dataset $dataset_dir \
    --pdb_dir_name $pdb_dir_name \
    --train_file train.csv \
    --valid_file valid.csv \
    --test_file test.csv \
    --metrics acc,mcc \
    --monitor valid/acc \
    --monitor_mode max \
    --c_alpha_max_neighbors $K \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --batch_token_num 12000 \
    --patience 5