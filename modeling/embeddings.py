import numpy as np
import os
import sys


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed


class AttributeEmbedding(nn.Module):
  def __init__(self, in_features, out_features, activation=None, norm=False):
    super(AttributeEmbedding, self).__init__()
    self.activation = activation
    self.norm = norm
    self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
    self.layer = TimeDistributed(self.layer)
    if self.norm:
      self.layer_norm = LayerNorm(out_features)

  def forward(self, inputs : Tensor) -> Tensor:
    outputs = self.layer(inputs) # [bs, seq, h]->[bs, seq, h]
    if self.norm:
      outputs = self.layer_norm(outputs)
    if self.activation == 'relu':
      outputs = F.relu(outputs)
    else: # TODO: involve more activation
      NotImplemented
    return outputs


class AttentionLayer(nn.Module):
  def __init__(self, config):
    super(AttentionLayer, self).__init__()
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

# G(x) * a + G(x) + bias
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

