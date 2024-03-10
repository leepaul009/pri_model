import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# from data import ArgoDataset, collate_fn
# from utils import gpu, to_long,  Optimizer, StepLR

# from layers import Conv1d, Res1d, Linear, LinearRes, Null
# from numpy import float64, ndarray
# from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from modeling.graph.layers import Linear
from modeling.neighborhoods import LocalNeighborhood


class MapNet(nn.Module):

  def __init__(self, hidden_dim, ):
    super(MapNet, self).__init__()
    # self.config = config
    # hidden_dim = config["hidden_dim"] # 128
    norm = "GN"
    ng = 1

    self.input = nn.Sequential(
      nn.Linear(20, hidden_dim),
      nn.ReLU(inplace=True),
      Linear(hidden_dim, hidden_dim, norm=norm, ng=ng, act=False),
    )
    self.relu = nn.ReLU(inplace=True)

    coordinates=['euclidian',]
    self.fuse = []
    self.edge = []
    self.norm = []
    self.ctr2 = []
    # self.local_neighborhood = []
    self.local_neighborhood = LocalNeighborhood(
      config, Kmax=16, coordinates=coordinates, 
      self_neighborhood=True, index_distance_max=8, nrotations=1)
      
    for i in range(4):
      self.fuse.append(
        nn.Linear(hidden_dim, hidden_dim, bias=False)
      )
      self.edge.append(
        nn.Linear(hidden_dim, hidden_dim, bias=False)
      )
      self.norm.append(nn.GroupNorm(gcd(ng, hidden_dim), hidden_dim))
      self.ctr2.append(Linear(hidden_dim, hidden_dim, norm=norm, ng=ng, act=False))
    self.fuse = nn.ModuleList(self.fuse)
    self.edge = nn.ModuleList(self.edge)
    self.norm = nn.ModuleList(self.norm)
    self.ctr2 = nn.ModuleList(self.ctr2)


  def forward(self, x, frames, ):
    # input (bs, seq, 20)
    bs_dim, seq_dim, input_dim = x.shape
    x = x.reshape(bs_dim * seq_dim, -1)
    x = self.input(x) # (bs*seq, 20) => (bs*seq, hidden)
    x = self.relu(x)

    # x = x.reshape(bs_dim, seq_dim, -1)

    res = x # (bs*seq, h)
    for i in range(4):
      # naming of node and edge has no meaning!!
      x_node = self.fuse[i](x)
      x_edge = self.edge[i](x)

      x_edge = x_edge.reshape(bs_dim, seq_dim, -1)
      input2localneighborhood = [frames, x_edge] # (bs, seq, h)
      output = self.local_neighborhood(input2localneighborhood)
      # (bs, seq, 16, 3), (bs, seq, 16, h)
      neighbor_coordinates, neighbors_attributes = output[0][0], output[1]
      tmp = torch.sum(neighbors_attributes, -2) # (bs, seq, h)
      tmp = tmp.reshape(bs_dim * seq_dim, -1) # (bs*seq, h)

      x = x_node + tmp

      x = self.norm[i](x)
      x = self.relu(x)

      x = self.ctr2[i](x)
      x += res
      x = self.relu(x)
      res = x
    
    x = x.reshape(bs_dim, seq_dim, -1) # (bs, seq, h)

    return x



