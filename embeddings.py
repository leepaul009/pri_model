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

def distance(coordinates1,coordinates2,squared=False,ndims=3):
  # D = (tf.expand_dims(coordinates1[...,0],axis=-1) - tf.expand_dims(coordinates2[...,0],axis=-2) )**2
  # for n in range(1,ndims):
  #     D += (tf.expand_dims(coordinates1[..., n], axis=-1) - tf.expand_dims(coordinates2[..., n], axis=-2)) ** 2
  # if not squared:
  #     D = tf.sqrt(D)
  D = ( coordinates1[...,0].unsqueeze(dim=-1) - coordinates2[...,0].unsqueeze(dim=-2) )**2
  for n in range(1,ndims):
    D += ( coordinates1[...,n].unsqueeze(dim=-1) - coordinates2[...,n].unsqueeze(dim=-2) )**2
  if not squared:
    D = torch.sqrt(D)
  return D

class LocalNeighborhood(nn.Module):
  def __init__(self, config,
      Kmax=10, coordinates=['index_distance'],
      self_neighborhood=True):
    super(LocalNeighborhood, self).__init__()
    self.config = config

    self.Kmax = Kmax
    self.coordinates = coordinates
    self.self_neighborhood = self_neighborhood

    if 'index_distance' in self.coordinates:
      self.first_format.append('index')
      self.second_format.append('index')


  def forward(self, inputs):
    if 'index' in self.first_format:
      first_index = inputs[self.first_format.index('index')]
    else:
      first_index = None
    if 'index' in self.second_format:
      if self.self_neighborhood:
        second_index = first_index
      else:
        second_index = inputs[len(self.first_format)+self.second_format.index('index')]
    else:
      second_index = None

    # list( [bs, s2, h] )
    second_attributes = inputs[-self.nattributes:]

    first_center = first_index.type_as(torch.float32)
    ndims = 1
    second_center = first_index.type_as(torch.float32)
    # [bs, s1, s2]
    distance_square = distance(first_center, second_center, squared=True,ndims=ndims)

    # neighbors = torch.unsqueeze(torch.argsort(distance_square)[:,:,:self.Kmax], dim=-1)
    neighbors = torch.argsort(distance_square)[:,:,:self.Kmax] # [bs, s1, k]
    # neighbors = neighbors.view(-1, self.Kmax) # TODO: continous?? [bs*s1, k]

    neighbors_attributes = []
    for attribute in second_attributes:
      attr_shape = attribute.shape
      # attribute = attribute.view(-1, attr_shape[-1]) # [bs*s2, h]
      # attribute[neighbors]
      
      neighbors_attr_batch = []
      for i in range(attr_shape[0]):
        # attribute = attribute.view(attr_shape[1], 1, attr_shape[-1])
        neighbors_attr = attribute[i][neighbors[i], :] # [s1, k, h]
        neighbors_attr_batch.append(neighbors_attr.unsqueeze(dim=0)) # [1, s1, k, h]
      neighbors_attr_batch = neighbors_attr_batch.concat(dim=0) # [bs, s1, k, h]
      neighbors_attributes.append(neighbors_attr_batch)
      



    return input