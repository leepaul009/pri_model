import numpy as np
import os
import sys


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from embeddings import EmbeddingOuterProduct

class NeighborhoodEmbedding(nn.Module):
  def __init__(self, config):
    super(NeighborhoodEmbedding, self).__init__()


  def forward(self, inputs):



    return inputs


class PriNet(nn.Module):
  def __init__(self, 
               config, 
               Lmax_aa=800,
               Lmax_atom=None,
               ):
    super(PriNet, self).__init__()
    if Lmax_atom is None:
      Lmax_atom = 9 * Lmax_aa
    # Lmax_nc, Lmax_nc_atom
    

  def forward(self, inputs):



    return inputs

