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
from modeling.esm.data import Alphabet
# post_preprocessor

from modeling.esm.data import Alphabet
from modeling.esm.esm2 import ESM2
from modeling.dbert.dbert import DBM
from modeling.dbert.config import DbmConfig


class DgLoss(nn.Module):
    def __init__(self):
        super(DgLoss, self).__init__()
        # self.config = config
        # self.loss = nn.MSELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

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


class GlobalNet(nn.Module):
    r"""
    It has two main components, sub graph and global graph.
    Sub graph encodes a polyline as a single vector.

    n_aa: number of aa
    n_atoms: number of atoms in one aa
    """

    def __init__(self, args_: utils.Args, esm2_cfg, alphabet: Alphabet):
        super(GlobalNet, self).__init__()
        global args
        args = args_
        self.hidden_size = args.hidden_size

        # self.esm2 = ESM2(
        #     num_layers=cfg.encoder_layers, # 6
        #     embed_dim=cfg.encoder_embed_dim, # 320
        #     attention_heads=cfg.encoder_attention_heads, # 20
        #     alphabet=alphabet,
        #     token_dropout=cfg.token_dropout, # True
        # )
        self.pwm_type = 'pssm'
        self.use_prot_chm = True

        self.esm2 = ESM2(
            num_layers=esm2_cfg.encoder_layers, # 6
            embed_dim=esm2_cfg.encoder_embed_dim, # 320
            attention_heads=esm2_cfg.encoder_attention_heads, # 20
            alphabet=alphabet,
            token_dropout=esm2_cfg.token_dropout, # True
        )
        
        config = DbmConfig(num_hidden_layers=6)
        self.dbm = DBM(config)
        
        
        self.loss = DgLoss()
        
        self.reg_head = nn.Linear(in_features=esm2_cfg.encoder_embed_dim + config.hidden_size, 
                                  out_features=self.hidden_size, bias=False)
        
        self.reg_pred = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        
        nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
        nn.init.normal_(self.reg_pred.weight, std=0.001)
        for l in [self.reg_head, self.reg_pred]:
            if l.bias is not None:
                nn.init.constant_(l.bias, 0)
                

    def forward(self, input: Tuple[List[Dict], Tensor, Tensor], device):
        mapping, aa_tokens, nc_tokens = input # aa_tokens (B,T), nc_tokens (B,T)
        ### amino acid features
        # aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(num_aa, 20)]
        # aa_pwm = utils.get_from_mapping(mapping, 'aa_pwm') if is_pwm_type_valid(self.pwm_type) else None
        # aa_chm = utils.get_from_mapping(mapping, 'aa_chm') if self.use_prot_chm else None

        ### nuc acid features
        # na_attributes = utils.get_from_mapping(mapping, 'nucleotide_attributes') # List[np.ndarray=(num_nc, 4)]
        # List[np.ndarray=(n_nc, 10)]
        # na_other_attributes = utils.get_from_mapping(mapping, 'nucleotide_other_attributes') if self.use_nc_chm else None

        batch_size = len(aa_tokens)
        aa_tokens, nc_tokens = aa_tokens.to(device), nc_tokens.to(device)
        
        aa_emb = self.esm2(aa_tokens) # (B, Tp, C=320)
        nc_emb = self.dbm(nc_tokens) # (B, Tn, C=768)

        assert aa_emb is not None
        assert nc_emb is not None
        
        aa_emb = aa_emb.permute(0,2,1).contiguous() # (B,C,T)
        nc_emb = nc_emb.permute(0,2,1).contiguous() # (B,C,T)
        
        aa_emb = F.avg_pool1d(aa_emb, aa_emb.shape[-1]).squeeze(-1) # (B,C)
        nc_emb = F.avg_pool1d(nc_emb, nc_emb.shape[-1]).squeeze(-1) # (B,C)
        
        comb_emb = torch.cat([aa_emb, nc_emb], dim=-1) # (B, Cp+Cn)
        
        comb_emb = self.reg_head(comb_emb) # (B, Cp+Cn) => (B, h)
        preds = self.reg_pred(comb_emb) # (B,h) => (B,1)
        
        targets = utils.get_from_mapping(mapping, 'label')
        targets = torch.tensor(targets, device=device, dtype=torch.float32).reshape(-1, 1) # (B,1)

        loss_output = self.loss(preds, targets) # (B,1) (B,1)

        return loss_output["loss"], preds, loss_output


