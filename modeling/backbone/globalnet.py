from typing import Dict, List, Tuple, NamedTuple, Any

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed
from modeling.loss import BinLoss, PredLoss
from modeling.decoder import Decoder
from axial_positional_embedding import AxialPositionalEmbedding
import utils

from modeling.layer import KMersNet, Conv2d
from modeling.sub_graphnet import SubGraph
from modeling.input_embedding import ProtEmbedding, NcEmbedding, \
                                     ProtDeepEmbedding, NcDeepEmbedding

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa, max_num_prot_chm_feats, \
    pwm_embeding_dims_by_type, is_pwm_type_valid

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics
# from modeling.esm.data import Alphabet
# post_preprocessor

from modeling.dbert.data import Alphabet as DAlphabet
from modeling.esm.data import Alphabet
from modeling.esm.esm2 import ESM2
from modeling.dbert.dbert import DBM
from modeling.dbert.config import DbmConfig


from modeling.batch_convert.mask import mask_tokens

class DgLoss(nn.Module):
    def __init__(self):
        super(DgLoss, self).__init__()
        # self.config = config
        self.reg_loss = nn.MSELoss(reduction="sum")
        # self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor):
        """
        pred=(N, 1)
        target=(N, 1)
        """
        output = dict()
        num_reg = pred.shape[0]
        loss = self.reg_loss(pred, target)
        output['num_reg'] = num_reg
        output['reg_loss'] = loss
        output['loss'] = loss / (num_reg + 1e-10)
        return output




class BasicHead(nn.Module):
    def __init__(self, 
                 aa_dim: int, 
                 nc_dim: int, 
                 hidden_dim: int, 
                 target_dim: int = 1
        ):
        super().__init__()
        
        self.nc_pool_kw = 2
        head_in_dim = aa_dim + int(nc_dim / self.nc_pool_kw)
        
        self.reg_head = nn.Linear(in_features=head_in_dim, 
                                  out_features=hidden_dim, 
                                  bias=False)
        self.reg_pred = nn.Linear(in_features=hidden_dim, 
                                  out_features=target_dim, 
                                  bias=False)
        
        nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
        nn.init.normal_(self.reg_pred.weight, std=0.001)
        for l in [self.reg_head, self.reg_pred]:
            if l.bias is not None:
                nn.init.constant_(l.bias, 0)
                
    def forward(self, 
                aa_emb: Tensor, 
                nc_emb: Tensor
        ):
        aa_emb = F.avg_pool2d(aa_emb, (aa_emb.shape[-2], 1)) # (B,T,C)=>(B,1,C=320)
        nc_emb = F.avg_pool2d(nc_emb, (nc_emb.shape[-2], self.nc_pool_kw)) # (B,T,C)=>(B,1,C/2=384)
        aa_emb = aa_emb.squeeze(-2) # (B,C)
        nc_emb = nc_emb.squeeze(-2) # (B,C)
        
        x = torch.cat([aa_emb, nc_emb], dim=-1) # (B, Cp+Cn)
        
        x = self.reg_head(x) # (B, Cp+Cn) => (B, h)
        preds = self.reg_pred(x) # (B,h) => (B,1)
        return preds


class InterActHead(nn.Module):
    def __init__(self, 
                 aa_dim: int, 
                 nc_dim: int, 
                 hidden_dim: int, 
                 target_dim: int = 1,
        ):
        super().__init__()
        self.fc1 = nn.Linear(in_features=aa_dim, 
                             out_features=hidden_dim, 
                             bias=False)
        self.layer_norm1 = LayerNorm(hidden_dim) ### use norm before TF layers
        
        self.fc2 = nn.Linear(in_features=nc_dim, 
                             out_features=hidden_dim, 
                             bias=False)
        self.layer_norm2 = LayerNorm(hidden_dim)
        
        ### use use_attention_decay
        self.global_graph = GlobalGraphRes(hidden_dim, use_attention_decay=True)
        
        ### use a smaller output dim 64
        self.fc3 = nn.Linear(in_features=hidden_dim, 
                              out_features=64, 
                              bias=False)
        ### use gelu and dropout
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(0.1)
        self.pred = nn.Linear(in_features=64, 
                              out_features=target_dim, 
                              bias=False)
        
    def forward(self,
                x1: Tensor, # (B, Tp, C=320)
                x2: Tensor, # (B, Tn, C=768)
                device: torch.device,
                lens1: List[int] = None,
                lens2: List[int] = None,
        ):

        batch_size  = x1.shape[0]
        # max_seq_num = x1.shape[1] + x2.shape[1] # Tp + Tn
        
        if lens1 is None or lens2 is None:
            lens1 = [len(x1[i]) for i in range(batch_size)]
            lens2 = [len(x2[i]) for i in range(batch_size)]
        
        x1 = self.fc1(x1)
        x1 = self.layer_norm1(x1)
        
        x2 = self.fc2(x2)
        x2 = self.layer_norm2(x2)
        
        complex_embedding = []
        for i in range(batch_size):
            cx = torch.cat([ x1[i, :lens1[i]], x2[i, :lens2[i]] ], dim=-2) # (Tp_i + Tn_i, H)
            complex_embedding.append(cx)
            
        complex_embedding, complex_lengths = utils.merge_tensors(complex_embedding, device=device)
        # assert complex_embedding.shape[1] == max_seq_num
        max_seq_num = max(complex_lengths)

        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device)
        for i, length in enumerate(complex_lengths):
            attention_mask[i, :length, :length].fill_(1)

        complex_embedding = self.global_graph(complex_embedding, attention_mask) # (B,T,C)
        
        complex_embedding = complex_embedding.permute(0,2,1).contiguous() # (B,C,T)
        complex_embedding = F.avg_pool1d(complex_embedding, complex_embedding.shape[-1]).squeeze(-1) # (B,C)
        
        complex_embedding = self.fc3(complex_embedding) # (B,C) => (B,C)
        complex_embedding = self.activation_fn(complex_embedding)
        complex_embedding = self.activation_dropout_module(complex_embedding)
        
        preds = self.pred(complex_embedding) # (B,C) => (B,1)
        
        return preds


vocab_by_kmers = {
    3: 69,
    4: 261,
    5: 1029,
    6: 4101,
}  

class GlobalNet(nn.Module):
    r"""
    It has two main components, sub graph and global graph.
    Sub graph encodes a polyline as a single vector.

    n_aa: number of aa
    n_atoms: number of atoms in one aa
    """
    def __init__(self, args_: utils.Args, esm2_cfg, alphabet: Alphabet, dalphabet: DAlphabet):
        super(GlobalNet, self).__init__()
        global args
        args = args_
        self.hidden_size = args.hidden_size
        
        self.alphabet  = alphabet
        self.dalphabet = dalphabet
        
        # self.pwm_type = 'pssm'
        # self.use_prot_chm = True

        ### set token_dropout False 
        self.esm2 = ESM2(
            num_layers = 6, # esm2_cfg.encoder_layers, # 6
            embed_dim=esm2_cfg.encoder_embed_dim, # 320
            attention_heads=esm2_cfg.encoder_attention_heads, # 20
            alphabet=alphabet,
            token_dropout=False, # esm2_cfg.token_dropout, # True
        )
        
        dbm_config = DbmConfig(
            vocab_size = vocab_by_kmers[args.kmers], 
            num_hidden_layers = 6, # 6,
        )
        self.dbm = DBM(dbm_config)
        
        self.loss = DgLoss()
        
        self.head_type = 'inter' # 'basic'
        
        if self.head_type == 'basic':
            self.head = BasicHead(esm2_cfg.encoder_embed_dim,
                                dbm_config.hidden_size,
                                self.hidden_size, 1)
        else:
            self.head = InterActHead(esm2_cfg.encoder_embed_dim,
                                dbm_config.hidden_size,
                                self.hidden_size, 1)
        # self.reg_head = nn.Linear(in_features=head_in_dim, 
        #                           out_features=self.hidden_size, bias=False)
        # self.reg_pred = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        
        # nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
        # nn.init.normal_(self.reg_pred.weight, std=0.001)
        # for l in [self.reg_head, self.reg_pred]:
        #     if l.bias is not None:
        #         nn.init.constant_(l.bias, 0)
                

    def forward(self, input: Tuple[List[Dict], Tensor, Tensor], device):
        mapping, aa_tokens, nc_tokens = input # aa_tokens (B,T), nc_tokens (B,T)
        
        ### aa pad_id=1, nc pad_id=0
        
        # aa_tokens = mask_tokens(aa_tokens, self.alphabet, is_extent=False)
        # nc_sequences = utils.get_from_mapping(mapping, 'nc_sequences')
        # print(nc_sequences)
        # nc_tokens = mask_tokens(nc_tokens, self.dalphabet)
        ### amino acid features
        # aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(num_aa, 20)]
        # aa_pwm = utils.get_from_mapping(mapping, 'aa_pwm') if is_pwm_type_valid(self.pwm_type) else None
        # aa_chm = utils.get_from_mapping(mapping, 'aa_chm') if self.use_prot_chm else None

        ### nuc acid features
        # na_attributes = utils.get_from_mapping(mapping, 'nucleotide_attributes') # List[np.ndarray=(num_nc, 4)]
        # List[np.ndarray=(n_nc, 10)]
        # na_other_attributes = utils.get_from_mapping(mapping, 'nucleotide_other_attributes') if self.use_nc_chm else None

        batch_size = len(aa_tokens)
        aa_seq_lens = []
        nc_seq_lens = []
        for i in range(batch_size):
            aa_seq_lens += [len( torch.where(aa_tokens[i] != 1)[0] )]
            nc_seq_lens += [len( torch.where(nc_tokens[i] != 0)[0] )]
        
        aa_tokens, nc_tokens = aa_tokens.to(device), nc_tokens.to(device)
        
        aa_emb = self.esm2(aa_tokens) # (B, Tp, C=320) normed
        nc_emb = self.dbm(nc_tokens) # (B, Tn, C=768) normed

        assert aa_emb is not None
        assert nc_emb is not None
        
        # aa_emb = aa_emb.permute(0,2,1).contiguous() # (B,C,T)
        # nc_emb = nc_emb.permute(0,2,1).contiguous() # (B,C,T)
        # aa_emb = F.avg_pool1d(aa_emb, aa_emb.shape[-1]).squeeze(-1) # (B,C)
        # nc_emb = F.avg_pool1d(nc_emb, nc_emb.shape[-1]).squeeze(-1) # (B,C)
        
        # aa_emb = F.avg_pool2d(aa_emb, (aa_emb.shape[-2], 1)) # (B,T,C)=>(B,1,C=320)
        # nc_emb = F.avg_pool2d(nc_emb, (nc_emb.shape[-2], self.nc_pool_kw)) # (B,T,C)=>(B,1,C/2=384)
        # aa_emb = aa_emb.squeeze(-2) # (B,C)
        # nc_emb = nc_emb.squeeze(-2) # (B,C)
        
        # comb_emb = torch.cat([aa_emb, nc_emb], dim=-1) # (B, Cp+Cn)
        
        if self.head_type == 'basic':
            preds = self.head(aa_emb, nc_emb) # (B,C,T)x2 => (B, Cp+Cn) => (B, h) => (B,1)
        else:
            preds = self.head(aa_emb, nc_emb, device, aa_seq_lens, nc_seq_lens)
        
        targets = utils.get_from_mapping(mapping, 'label')
        targets = torch.tensor(targets, device=device, dtype=torch.float32).reshape(-1, 1) # (B,1)

        loss_output = self.loss(preds, targets) # (B,1) (B,1)

        return loss_output["loss"], preds, loss_output


