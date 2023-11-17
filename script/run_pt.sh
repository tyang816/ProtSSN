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