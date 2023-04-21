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
    self.epsilon = 1e-6

    self.Lmax = 256
    self.Kmax = 14
    self.nfeatures_graph = 1
    self.nheads = 64
    self.nfeatures_output = 1


  def forward(self, inputs):
    attention_coefficients, node_outputs, graph_weights = inputs
    device = attention_coefficients.device

    epsilon = torch.tensor(self.epsilon).to(device)

    # [N, aa_seq, k, 1, 64]
    attention_coefficients = attention_coefficients.reshape(
      -1, self.Lmax, self.Kmax, self.nfeatures_graph, self.nheads)
    # [N, aa_seq, k, 1, 64]
    node_outputs = node_outputs.reshape(
      -1, self.Lmax, self.Kmax, self.nfeatures_output, self.nheads)

    # [N, aa_seq, k, 1, 64] => [N, aa_seq, k, 1, 64]
    # attention_coefficients -= tf.reduce_max(
    #     attention_coefficients, axis=[-3, -2], keep_dims=True)
    attention_coefficients -= torch.sum(
      attention_coefficients, dim=[-3, -2], keepdim=True)
    # wt[N, aa_seq, k, 1, 1]*att[N, aa_seq, 1, 1, 64] => [N, aa_seq, k, 64]
    # attention_coefficients_final = tf.reduce_sum(tf.expand_dims(
    #     graph_weights, axis=-1) * K.exp(attention_coefficients), axis=-2)
    attention_coefficients_final = torch.sum(
      graph_weights.unsqueeze(dim=-1) * torch.exp(attention_coefficients), dim=-2)
    # [N, aa_seq, k, 64] / [N, aa_seq, 1, 64] + eps
    # attention_coefficients_final /= tf.reduce_sum(
    #     tf.abs(attention_coefficients_final), axis=-2, keep_dims=True) + self.epsilon
    attention_coefficients_final /= torch.sum(
      torch.abs(attention_coefficients_final), dim=-2, keepdim=True) + epsilon

    # here: max_pool=>[64], wt+exp=>[k,64]
    # [N, aa_seq, k, 1, 64]*[N, aa_seq, k, 1, 64]=>[N,s,k,1,64]=reduce_sum(k)=>[N,s,1,64]=>[N,s,64]
    # output_final = tf.reshape(tf.reduce_sum(node_outputs * tf.expand_dims(
    #     attention_coefficients_final, axis=-2), axis=2), [-1, self.Lmax, self.nfeatures_output * self.nheads])
    output_final = node_outputs * attention_coefficients_final.unsqueeze(dim=-2)
    output_final = torch.sum(output_final, dim=-2).reshape(-1, self.Lmax, self.nfeatures_output * self.nheads)

    return [output_final, attention_coefficients_final]


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

  # inputs: indices1[N,s1,1], indices2[N,s2,1], attr1[N,s1,h], attr2[N,s2,h]
  # outputs:
  def forward(self, inputs):
    if 'index' in self.first_format:
      # [bs, s1, 1]
      first_index = inputs[self.first_format.index('index')]
    else:
      first_index = None
    if 'index' in self.second_format:
      if self.self_neighborhood:
        second_index = first_index
      else:
        # [bs, s2, 1]
        second_index = inputs[len(self.first_format)+self.second_format.index('index')]
    else:
      second_index = None

    assert first_index.shape[0] == second_index.shape[0]
    batch_size = first_index.shape[0]

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
      
      assert attr_shape[0] == batch_size
      neighbors_attr_batch = []
      for i in range(batch_size):
        # attribute = attribute.view(attr_shape[1], 1, attr_shape[-1])
        neighbors_attr = attribute[i, neighbors[i], :] # [s1, k, h]
        neighbors_attr_batch.append(neighbors_attr.unsqueeze(dim=0)) # [1, s1, k, h]
      neighbors_attr_batch = neighbors_attr_batch.concat(dim=0) # [bs, s1, k, h]
      neighbors_attributes.append(neighbors_attr_batch)
      
    neighbor_coordinates = []
    
    neighbor_second_indices = []
    for i in range(batch_size):
      # [s2, 1] => [s1, k, 1] => [1, s1, k, 1]
      neighbor_second_indices.append(
        second_index[i, neighbors[i], :].unsqueeze(dim=0) )
    neighbor_second_indices = neighbor_second_indices.concat(dim=0) # [bs, s1, k, 1]

    index_distance = first_index.unsqueeze(dim=-2) - neighbor_second_indices
    index_distance = torch.abs(index_distance.type_as(torch.float32))
    neighbor_coordinates.append(index_distance)

    output = [neighbor_coordinates] + neighbors_attributes
    return output