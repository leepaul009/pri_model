from typing import Dict, List, Tuple, NamedTuple, Any

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed
from modeling.loss import BinLoss, PredLoss
from modeling.decoder import Decoder
from axial_positional_embedding import AxialPositionalEmbedding
import utils

from modeling.layer import KMersNet, Conv2d
from modeling.sub_graphnet import SubGraph

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa, max_num_prot_chm_feats, \
    pwm_embeding_dims_by_type, is_pwm_type_valid




class ProtEmbedding(nn.Module):
  def __init__(self, args, aa_emb_hidden_dim):
    super(ProtEmbedding, self).__init__()
    self.args = args
    hidden_size = args.hidden_size
    self.hidden_size = hidden_size

    # sub_graph_hidden_dim = hidden_size // 2 if self.use_atom_graph else hidden_size
        
    self.pwm_type = args.pwm_type
    self.use_prot_chm = args.use_prot_chm_feature
    max_seq_len = 6000

    ### update output dim
    # aa_emb_hidden_dim = sub_graph_hidden_dim if self.use_sub_graph else hidden_size
    self.aa_out_dim = aa_emb_hidden_dim
    if is_pwm_type_valid(self.pwm_type):
        self.aa_out_dim = aa_emb_hidden_dim // 2
        print("use prot pwm feature, output dim = {}".format(self.aa_out_dim))
    if self.use_prot_chm:
        self.aa_out_dim = aa_emb_hidden_dim // 2
        print("use prot chm feature, output dim = {}".format(self.aa_out_dim))
    
    ### seq embedding
    self.aa_input_features = 20
    self.aa_seq_layer = nn.Linear(in_features=self.aa_input_features, 
                                  out_features=self.aa_out_dim, 
                                  bias=False)
    self.aa_seq_layer = TimeDistributed(self.aa_seq_layer)
    # position embedding
    self.aa_pos_emb = AxialPositionalEmbedding(
        dim = self.aa_out_dim,
        axial_shape = (math.ceil(max_seq_len / 64), 64))
    self.na_seq_norm = LayerNorm(self.aa_out_dim)

    ### pwm embedding
    if is_pwm_type_valid(self.pwm_type):
        self.aa_pwm_layer = nn.Linear(
            in_features=pwm_embeding_dims_by_type[self.pwm_type], 
            out_features=self.aa_out_dim, 
            bias=False)
        self.aa_pwm_layer = TimeDistributed(self.aa_pwm_layer)
        # position embedding
        self.aa_pwm_pos_emb = AxialPositionalEmbedding(
            dim = self.aa_out_dim,
            axial_shape = (math.ceil(max_seq_len / 64), 64))
        self.na_pwm_norm = LayerNorm(self.aa_out_dim)
    
    ################ prot chemistry layer ################
    if self.use_prot_chm:
        self.prot_chm_dim = max_num_prot_chm_feats
        self.prot_chm_layer = nn.Linear(in_features=self.prot_chm_dim, 
                                        out_features=self.aa_out_dim, 
                                        bias=False)
        self.prot_chm_layer = TimeDistributed(self.prot_chm_layer)
        # position embedding
        self.aa_chm_pos_emb = AxialPositionalEmbedding(
            dim = self.aa_out_dim,
            axial_shape = (math.ceil(max_seq_len / 64), 64))
        self.na_chm_norm = LayerNorm(self.aa_out_dim)

  def aa_attribute_embeding(
        self,
        aa_attributes : List[np.ndarray], 
        aa_pwm : List[np.ndarray],
        aa_chm : List[np.ndarray],
        device, batch_size) -> Tuple[Tensor, List[int]]:
    """
    compute embedding of amino acid
    inputs:
        aa_attributes: List[Tensor=(n_aa, 20)]
    outputs:
        Tensor=(N, max(num_aa), h/2)
        List[int], each prot length
    """
    prot_list = []
    for i in range(batch_size):
        prot = torch.tensor(aa_attributes[i], device=device)
        prot_list.append(prot)
    x, lengths = utils.merge_tensors(prot_list, device=device) # [bs, max_n_aa, 20]
    x = self.aa_seq_layer(x) # [bs, max_n_aa, 20]->[bs, max_n_aa, h/4]
    x += self.aa_pos_emb(x)
    # x = self.dropout(x)
    x = self.na_seq_norm(x)
    
    other_embedding = None
    if is_pwm_type_valid(self.pwm_type) and aa_pwm is not None:
        pwm_list = []
        for i in range(batch_size):
            tensor = torch.tensor(aa_pwm[i], device=device)
            pwm_list.append(tensor)
        aa_pwm_embedding, _ = utils.merge_tensors(pwm_list, device=device) # [bs, max_n_aa, 30]
        aa_pwm_embedding = self.aa_pwm_layer(aa_pwm_embedding) # [bs, max_n_aa, 30]->[bs, max_n_aa, h/4]
        aa_pwm_embedding += self.aa_pwm_pos_emb(aa_pwm_embedding)
        # aa_pwm_embedding = self.dropout(aa_pwm_embedding)
        aa_pwm_embedding = self.na_pwm_norm(aa_pwm_embedding)
        # aa_pwm_embedding = F.relu(aa_pwm_embedding)
        # x = torch.cat([x, aa_pwm_embedding], dim=-1) # [bs, max_n_aa, h/2]
        other_embedding = aa_pwm_embedding

    if self.use_prot_chm:
        chm_list = []
        for i in range(batch_size):
            tensor = torch.tensor(aa_chm[i], device=device)
            chm_list.append(tensor)
        chm_embedding, _ = utils.merge_tensors(chm_list, device=device) # (bs, max(num_aa), 37)
        chm_embedding = self.prot_chm_layer(chm_embedding) # (bs, max(num_aa), h/4)
        chm_embedding += self.aa_chm_pos_emb(chm_embedding)
        # chm_embedding = self.dropout(chm_embedding)
        chm_embedding = self.na_chm_norm(chm_embedding)
        # chm_embedding = F.relu(chm_embedding)
        # x = torch.cat([x, chm_embedding], dim=-1) # (bs, max(num_aa), h/2)
        if other_embedding is not None:
            other_embedding += chm_embedding
        else:
            other_embedding = chm_embedding
    
    if other_embedding is not None:
        x = torch.cat([x, other_embedding], dim=-1)
    
    x = F.relu(x)
    return x, lengths

  def forward(self, aa_attributes, aa_pwm, aa_chm, device, batch_size):
    
    aa_embedding, prot_lengths = self.aa_attribute_embeding(aa_attributes, aa_pwm, aa_chm, device, batch_size)
        
    return aa_embedding, prot_lengths


class NcEmbedding(nn.Module):
  def __init__(self, args, nc_emb_hidden_dim):
    super(NcEmbedding, self).__init__()
    self.use_nc_chm = args.use_chemistry
    self.use_kmers = False
    self.use_conv =  True
    na_max_seq_len = 1024

    self.nc_out_dim = nc_emb_hidden_dim
    if self.use_nc_chm:
        self.nc_out_dim = nc_emb_hidden_dim // 2

    if self.use_kmers:
        self.kmers_net = KMersNet(base_channel=8, out_channels=self.nc_out_dim)
    elif self.use_conv: ## TODO: fix as current not allowed
        self.na_conv2d = Conv2d(1, 8, 
                                kernel_size=(3, 5),
                                bn=True, same_padding=True)
        self.na_fc = nn.Linear(in_features=32, out_features=self.nc_out_dim, bias=False) 
    else: ## TODO: fix as current not allowed
        self.nc_seq_layer = nn.Linear(in_features=4, out_features=self.nc_out_dim, bias=False)
        self.nc_seq_layer = TimeDistributed(self.nc_seq_layer)

    self.na_norm = LayerNorm(self.nc_out_dim)
    self.na_pos_emb = AxialPositionalEmbedding(
        dim = self.nc_out_dim,
        axial_shape = (math.ceil(na_max_seq_len / 64), 64))

    if self.use_nc_chm:
        # (num_nc, 1+9) => (num_nc, h/2)
        self.nc_chm_dim = 10
        self.nc_chm_layer = nn.Linear(in_features=self.nc_chm_dim, 
                                      out_features=self.nc_out_dim, 
                                      bias=False)
        # self.na_comb_layer = nn.Linear(in_features=hidden_size, out_features=sub_graph_hidden_dim, bias=False)
        # concat seq_feat and other_feat to (num_nc, h) => (num_nc, h/2)
        # nn.init.kaiming_uniform_(self.nc_chm_layer.weight, a=1)
        # nn.init.kaiming_uniform_(self.na_comb_layer.weight, a=1)

  def na_attribute_embeding(
          self,
          na_attributes : List[np.ndarray], 
          na_other_attributes : List[np.ndarray], 
          device, batch_size) -> Tuple[Tensor, List[int]]:
      """
      compute embedding of Nucleic Acids
      inputs:
          na_attributes: List[Tensor=(n_na, 4)]
      outputs:
          List[Tensor=(n_na, h/2)] 
      """
      input_list = []
      for i in range(batch_size):
          tensor = torch.tensor(na_attributes[i], device=device)
          input_list.append(tensor)
      x, lengths = utils.merge_tensors(input_list, device=device) # (bs, max(n_na), 4)

      if self.use_kmers:
          x = self.kmers_net(x.unsqueeze(1)) # (bs, 1, max(n_na), 4)->(bs, max(n_na), h)
      elif self.use_conv:
          x = self.na_conv2d(x.unsqueeze(1)) # (N,1,seq,4)=>(N,C,seq,4)
          x = x.permute(0,2,1,3).contiguous() # (N,Seq,C,4)
          x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # (N,Seq,C*4)
          x = self.na_fc(x) # (N,Seq,h)
          
      else:
          x = self.nc_seq_layer(x) # (bs, max(n_na), 4)->(bs, max(n_na), h)
      x += self.na_pos_emb(x)

      if self.use_nc_chm and na_other_attributes is not None:
          chm_list = []
          for i in range(batch_size):
              tensor = torch.tensor(na_other_attributes[i], device=device)
              chm_list.append(tensor)
          chm_emb, _ = utils.merge_tensors(chm_list, device=device) # (bs, max(n_na), 10)
          chm_emb = self.nc_chm_layer(chm_emb) # (bs, max(n_na), h/2)
          x = torch.cat([x, chm_emb], dim=-1) # (bs, max(n_na), h)
      
      x = self.na_norm(x) # (N,Seq,h)
      x = F.relu(x)
      return x, lengths

  def forward(self, na_attributes, na_other_attributes, device, batch_size):
    na_embedding, nc_lengths = self.na_attribute_embeding(na_attributes, na_other_attributes, device, batch_size)
    return na_embedding, nc_lengths


class ProtDeepEmbedding(nn.Module):
  def __init__(self, args, out_dim):
    super(ProtDeepEmbedding, self).__init__()
    in_dim = args.prot_emb_size
    max_seq_len = 6000

    self.aa_seq_layer = nn.Linear(in_features=in_dim, 
                                  out_features=out_dim, 
                                  bias=False)
    self.aa_seq_layer = TimeDistributed(self.aa_seq_layer)
    # position embedding
    self.aa_pos_emb = AxialPositionalEmbedding(
        dim = out_dim,
        axial_shape = (math.ceil(max_seq_len / 64), 64))
    self.seq_norm = LayerNorm(out_dim)

  def forward(self, aa_attributes, aa_pwm, aa_chm, device, batch_size):
    prot_list = []
    for i in range(batch_size):
      prot = torch.tensor(aa_attributes[i], device=device)
      prot_list.append(prot)
    x, lengths = utils.merge_tensors(prot_list, device=device) # [bs, max_n_aa, in]
    x = self.aa_seq_layer(x) # [bs, max_n_aa, in]->[bs, max_n_aa, h]
    x += self.aa_pos_emb(x)
    x = self.seq_norm(x)
    x = F.relu(x)
    return x, lengths

class NcDeepEmbedding(nn.Module):
  def __init__(self, args, out_dim):
    super(NcDeepEmbedding, self).__init__()
    in_dim = args.nc_emb_size
    max_seq_len = 1024

    self.seq_layer = nn.Linear(in_features=in_dim, 
                                  out_features=out_dim, 
                                  bias=False)
    self.seq_layer = TimeDistributed(self.seq_layer)
    # position embedding
    self.pos_emb = AxialPositionalEmbedding(
        dim = out_dim,
        axial_shape = (math.ceil(max_seq_len / 64), 64))
    self.seq_norm = LayerNorm(out_dim)

  def forward(self, na_attributes, na_other_attributes, device, batch_size):
    input_list = []
    for i in range(batch_size):
      tensor = torch.tensor(na_attributes[i], device=device)
      input_list.append(tensor)
    x, lengths = utils.merge_tensors(input_list, device=device) # (bs, max(n_na), 4)
    x = self.seq_layer(x) # [bs, max_n_aa, in]->[bs, max_n_aa, h]
    x += self.pos_emb(x)
    x = self.seq_norm(x)
    x = F.relu(x)
    return x, lengths