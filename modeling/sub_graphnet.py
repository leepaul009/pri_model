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

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa, max_num_prot_chm_feats, \
    pwm_embeding_dims_by_type, is_pwm_type_valid

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics


class SubGraph(nn.Module):
    # config:
    #   depth:int, hidden_size:int, point_level-4-3:bool
    def __init__(self, config, hidden_size, depth):
        super(SubGraph, self).__init__()
        
        # depth = config['depth']
        # hidden_size = config['hidden_size']
        self.hidden_size = hidden_size
        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) 
                                     for _ in range(depth)])
        
        # self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        if config['use_batch_norm']:
            self.layers_2 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(depth)])
        else:
            self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            
        #if config['point_level-4-3']:
        # self.layer_0_again = MLP(hidden_size)
        
        if config['use_atom_embedding']:
            atom_nfeatures = config['atom_nfeatures']
            self.atom_emb_layer = nn.Embedding(num_embeddings=atom_nfeatures + 1, 
                                               embedding_dim=atom_nfeatures)
            max_seq_len = max_num_atoms_in_aa + 5
            self.atom_pos_emb = AxialPositionalEmbedding(
                dim = atom_nfeatures,
                axial_shape = (math.ceil(max_seq_len / 64), 64))

            self.layer_0 = MLP(atom_nfeatures, hidden_size)
        
        self.config = config
        self.use_dropout = True
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

    # input_list:
    #   case1: a chain, shape=residues*[atoms, hidden] (att along atoms)
    #   case2: a batch of chains, shape=N*[residues, hidden] (att along residues)
    # def forward1(self, input_list : List[Tensor]):
    def forward(self, hidden_states : Tensor, lengths: List[int]):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        # input_list is list of tensor[nodes(atom/aa), hidden]
        # merge with padding: hidden_states = [lines, max_nodes, hidden]
        # hidden_states, lengths = utils.merge_tensors(input_list, device) # [aa, max_atoms, h]
        # hidden_states = input.unsqueeze(-1) # (aa, max(num_atoms), h)
        # hidden_size = hidden_states.shape[2] # here hidden size is input's
        max_vector_num = hidden_states.shape[1] # do att along this dim

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device) # atom case: [aa, max_atoms, max_atoms]
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            if hidden_states.type is not torch.long:
                hidden_states = hidden_states.type(torch.long) # (aa, max(num_atoms))
            hidden_states = self.atom_emb_layer(hidden_states) # (aa, max(num_atoms), e)
            hidden_states += self.atom_pos_emb(hidden_states)

        hidden_states = self.layer_0(hidden_states)
        # hidden_states = self.layer_0_again(hidden_states) # (aa, max(num_atoms), h/2)
        
        # mask sequence dim as each item in batch has difference seq length
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask)
            if self.use_dropout:
                hidden_states = self.dropout(hidden_states)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            
            if self.config['use_batch_norm']:
                orig_shape = hidden_states.shape
                hidden_states = self.layers_2[layer_index](hidden_states.reshape(-1, orig_shape[-1]))
                hidden_states = hidden_states.reshape(orig_shape)
            else:         
                hidden_states = self.layers_2[layer_index](hidden_states)
        
        if self.config['use_atom_embedding']:
            hidden_states = torch.max(hidden_states, dim=1)[0] # (aa, h)
        return hidden_states

    def forward2(self, input : Tensor, lengths: Tensor):
        device = input.device
        batch_size = input.shape[0]
        max_vector_num = input.shape[1]

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device)
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            if input.type is not torch.long:
                input = input.type(torch.long).squeeze(-1) # (N, seq,1) -> (N,seq,)
            hidden_states = self.atom_emb_layer(input)
            hidden_states += self.atom_pos_emb(hidden_states)

        hidden_states = self.layer_0(hidden_states)

        # if self.config['point_level-4-3']:
        hidden_states = self.layer_0_again(hidden_states)
        
        # mask sequence dim as each item in batch has difference seq length
        for i in range(batch_size):
            # assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        # list_hidden_states = []
        # for i in range(batch_size):
        #     iend = lengths[i]
        #     iend = max(1, iend)
        #     hidden_states_per_aa = torch.max(hidden_states[i, :iend], dim=0)[0]
        #     list_hidden_states.append(hidden_states_per_aa.unsqueeze(0))
        # hidden_states = torch.cat(list_hidden_states)
        
        return torch.max(hidden_states, dim=1)[0]
        # return hidden_states
        # return torch.max(hidden_states, dim=1)[0], \
        #        torch.cat(utils.de_merge_tensors(hidden_states, lengths))


