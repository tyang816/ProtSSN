H=512
K=20
python protssn.py \
    --gnn_hidden_dim $H \
    --gnn_model_path model/protssn_k"$K"_h"$H".pt \
    --c_alpha_max_neighbors $K \
    --pdb_file data/sol/esmfold_pdb/protein_1.ef.pdb