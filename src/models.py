import torch
import random
import gc
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from torch_geometric.data import Batch, Dataset
from transformers import AutoTokenizer, EsmModel
from typing import *
from src.module.egnn.network import EGNN
from src.module.gcn.network import GCN
from src.module.gat.network import GAT

class PLM_model(nn.Module):
    possible_amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
    one_letter = {
        'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
        'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
        'GLY':'G', 'PRO':'P', 'CYS':'C'
        }
    
    def __init__(self, args):
        super().__init__()
        # load global config
        self.args = args
        
        # esm on the first cuda
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.plm)
        self.model = EsmModel.from_pretrained(self.args.plm).cuda()
        
        
    def forward(self, batch):
        with torch.no_grad():
            if not isinstance(batch, List):
                batch = [batch]
            
            if not hasattr(self.args, "noise_type"):
                one_hot_seqs = [list(elem.x[:,:20].argmax(1)) for elem in batch]
                truth_res_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_seqs]
                input_seqs = truth_res_seqs
            
            elif self.args.noise_type == 'mask':
                one_hot_truth_seqs = [elem.y for elem in batch]
                truth_res_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_truth_seqs]
                input_seqs = self._mask_input_sequence(truth_res_seqs)
            
            elif self.args.noise_type == 'mut':
                one_hot_seqs = [list(elem.x[:,:20].argmax(1)) for elem in batch]
                muted_res_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_seqs]
                input_seqs = muted_res_seqs
            else:
                raise ValueError(f"No implement of {self.args.noise_type}")

            batch_graph = self._nlp_inference(input_seqs, batch)
        return batch_graph
        
    @torch.no_grad()
    def _mask_input_sequence(self, truth_res_seqs):
        input_seqs = []
        self.mask_ratio = self.args.noise_ratio
        for truth_seq in truth_res_seqs:
            masked_seq = ""
            for truth_token in truth_seq:
                pattern = torch.multinomial(torch.tensor([1 - self.args.noise_ratio, 
                                                          self.mask_ratio*0.8, 
                                                          self.mask_ratio*0.1, 
                                                          self.mask_ratio*0.1]), 
                                            num_samples=1,
                                            replacement=True)
                # 80% of the time, we replace masked input tokens with mask_token ([MASK])
                if pattern == 1:
                    masked_seq += '<mask>'
                # 10% of the time, we replace masked input tokens with random word
                elif pattern == 2:
                    masked_seq += random.sample(list(self.one_letter.values()), 1)[0]
                # The rest of the time (10% of the time) we keep the masked input tokens unchanged
                else:
                    masked_seq += truth_token
            input_seqs.append(masked_seq)
        return input_seqs
    
    
    @torch.no_grad()
    def _nlp_inference(self, input_seqs, batch):    
        inputs = self.tokenizer(input_seqs, return_tensors="pt", padding=True).to("cuda:0")
        batch_lens = (inputs["attention_mask"] == 1).sum(1) - 2
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        for idx, (hidden_state, seq_len) in enumerate(zip(last_hidden_states, batch_lens)):
            batch[idx].esm_rep = hidden_state[1: 1+seq_len]
            del batch[idx].seq
                
        # move to the GNN devices
        batch = [elem.cuda() for elem in batch]
        batch_graph = Batch.from_data_list(batch)
        gc.collect()
        torch.cuda.empty_cache()
        return batch_graph



class GNN_model(nn.Module):    
    def __init__(self, args):
        super().__init__()
        # load graph network config which usually not change
        self.gnn_config = args.gnn_config
        # load global config
        self.args = args
        
        # calculate input dim according to the input feature
        self.out_dim = 20
        self.input_dim = self.args.plm_hidden_size
        
        # gnn on the rest cudas
        if "egnn" == self.args.gnn:
            self.GNN_model = EGNN(self.gnn_config, self.args, self.input_dim, self.out_dim)
        elif "gcn" == self.args.gnn:
            self.GNN_model = GCN(self.gnn_config, self.input_dim, self.out_dim)
        elif "gat" == self.args.gnn:
            self.GNN_model = GAT(self.gnn_config,self.input_dim, self.out_dim)
        else:
            raise KeyError(f"No implement of {self.opt['gnn']}")
        self.GNN_model = self.GNN_model.cuda()

    def forward(self, batch_graph):
        gnn_out = self.GNN_model(batch_graph)
        return gnn_out


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out

class Attention1dPoolingProjection(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.25) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final(x)
        return x

class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.25
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(hidden_size)
        self.attention1d_projection = Attention1dPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, x, input_mask=None):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x

class MeanPooling(nn.Module):
    """Mean Pooling for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            # Applying input_mask to zero out masked values
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class MeanPoolingProjection(nn.Module):
    """Mean Pooling with a projection layer for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, mean_pooled_features):
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = ACT2FN['gelu'](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MeanPoolingHead(nn.Module):
    """Mean Pooling Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, dropout=0.25):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.mean_pooling_projection = MeanPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, features, input_mask=None):
        mean_pooling_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.mean_pooling_projection(mean_pooling_features)
        return x


class LightAttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size=1280, num_labels=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttentionPoolingHead, self).__init__()

        self.feature_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, num_labels)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, sequence_length, hidden_size] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,num_labels] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, sequence_length]
        o = self.feature_convolution(x)  # [batch_size, hidden_size, sequence_length]
        o = self.dropout(o)  # [batch_gsize, hidden_size, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, hidden_size, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, hidden_size]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, hidden_size]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*hidden_size]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, num_labels]


class ProtssnClassification(nn.Module):
    def __init__(self, args, plm_model, gnn_model):
        super().__init__()
        self.args = args
        self.plm_model = plm_model
        self.gnn_model = gnn_model
        if args.pooling_method == "mean":
            self.pooling_head = MeanPoolingHead(args.plm_hidden_size, args.num_labels, args.pooling_dropout)
        elif args.pooling_method == "attention1d":
            self.pooling_head = Attention1dPoolingHead(args.plm_hidden_size, args.num_labels, args.pooling_dropout)
        elif args.pooling_method == "light_attention":
            self.pooling_head = LightAttentionPoolingHead(args.plm_hidden_size, args.num_labels, args.pooling_dropout)
        else:
            raise KeyError(f"No implement of {args.pooling_method}")
        
    def forward(self, batch):
        with torch.no_grad():
            batch_graph = self.plm_model(batch)
            _, embeds = self.gnn_model(batch_graph)
        
        graph_sizes = torch.unique(batch_graph.batch, return_counts=True)[1]  
        max_nodes = graph_sizes.max().item()
        batch_size = len(graph_sizes)
        padded_embeds = torch.zeros(batch_size, max_nodes, embeds.shape[-1]).to(embeds.device)
        attention_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool).to(embeds.device)
        start_idx = 0  
        for i, size in enumerate(graph_sizes):  
            end_idx = start_idx + size  
            padded_embeds[i, :size] = embeds[start_idx:end_idx]  
            attention_mask[i, :size] = True  
            start_idx = end_idx
        
        out = self.pooling_head(padded_embeds, attention_mask)
        
        return out