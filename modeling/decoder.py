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
from axial_positional_embedding import AxialPositionalEmbedding
import utils

from modeling.layer import KMersNet

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics





class Decoder(nn.Module):

  def __init__(self, args_: utils.Args):
      super(Decoder, self).__init__()



  def forward(self, 
              # mapping: List[Dict], 
              # batch_size, 
              # lane_states_batch: List[Tensor], 
              # inputs: Tensor,
              # inputs_lengths: List[int], 
              # hidden_states: Tensor, 
              gt_logics,   # (bs, 1) int
              gt_delta,    # (bs, 1)
              bin_ctrs,    # (bs, num_bins)
              bin_half_w,  # (bs, num_bins)
              pred_logics, # (bs, num_bins)
              pred_delta,  # (bs, 1)
              ):
    ###
    device = pred_logics.device
    batch_size = pred_logics.shape[0]
    
      # offset = (label - ctr) / width
    # label = offset * width + ctr
    
    # gt_logics = gt_logics.reshape(-1) # (bs,)
    # ind1 = torch.arange(gt_logics.shape[0]).to(device) # (bs,)

    # ctr   = bin_ctrs[ind1, gt_logics].reshape(-1, 1) # (bs, 1)
    # width = bin_half_w[ind1, gt_logics].reshape(-1, 1) # (bs, 1)

    # gt_pred = gt_delta * width + ctr # (bs, 1)
    
    # pred_logics = pred_logics # (bs,)
    pred_logics = torch.argmax(pred_logics, dim=-1, keepdim=False) # (bs, num_cls) -> (bs,)
    ind1 = torch.arange(pred_logics.shape[0]).to(device) # (bs,)
    
    ctr   = bin_ctrs[ind1, pred_logics].reshape(-1, 1) # (bs, 1)
    width = bin_half_w[ind1, pred_logics].reshape(-1, 1) # (bs, 1)
    
    pred = pred_delta * width + ctr # (bs, 1)

    return pred