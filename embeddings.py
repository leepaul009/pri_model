import numpy as np
import os
import sys


import torch
from torch import Tensor, nn
from torch.nn import functional as F




class EmbeddingOuterProduct(nn.Module):
  def __init__(self, config):
    super(EmbeddingOuterProduct, self).__init__()
    self.config = config

    self.sum_axis = 2
    self.use_bias = False
    self.kernel12 = torch.nn.Parameter(data=torch.Tensor(32, 12, 128), requires_grad=True)
    self.kernel1 = torch.nn.Parameter(data=torch.Tensor(32, 128), requires_grad=True)
    self.bias = torch.nn.Parameter(data=torch.Tensor(128), requires_grad=True)
    # init

  def forward(self, inputs):
    first_input = inputs[0] # [bs, seq, k, feat]
    second_input = inputs[1] # [bs, seq, k, feat]

    if self.sum_axis is not None:
      temp = torch.unsqueeze(first_input, dim=-1) \
            * torch.unsqueeze(second_input, dim=-2)
      outer_product = torch.sum(temp, dim=self.sum_axis)

    activity = torch.tensordot(outer_product, self.kernel12, dims=([-2, -1], [0, 1]))

    activity += torch.tensordot(first_input.sum(dim=self.sum_axis), self.kernel1, ([-1],[0]))
    if self.use_bias:
      activity += self.bias.reshape(1, 1, -1)
    return activity


