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


class GaussianKernel(nn.Module):
  def __init__(self, d, N, initial_values, covariance_type='diag', eps=1e-1, **kwargs):
    super(GaussianKernel, self).__init__(**kwargs)
    self.support_masking = True
    self.eps = eps
    self.N = N # 32 for example
    self.initial_values = initial_values
    self.covariance_type = covariance_type
    assert self.covariance_type in ['diag', 'full']

    ## build
    self.d = d
    self.center_shape = torch.Size([self.d, self.N])
    self.kernel_centers = torch.nn.Parameter(data=torch.Tensor(self.center_shape), requires_grad=True) # (d,N)
    self.kernel_widths = None
    self.sqrt_precision = None

    if self.covariance_type == 'diag':
      self.width_shape = torch.Size([self.d, self.N])
      self.kernel_widths = torch.nn.Parameter(data=torch.Tensor(self.width_shape), requires_grad=True) # (d,N)

    elif self.covariance_type == 'full':
      self.sqrt_precision_shape = torch.Size([self.d, self.d, self.N])
      self.sqrt_precision = torch.nn.Parameter(data=torch.Tensor(self.sqrt_precision_shape), requires_grad=True) # (d, d,N)

    nn.init.kaiming_normal_(self.kernel_centers, mode="fan_in", nonlinearity="relu")
    if self.kernel_widths is not None:
      nn.init.kaiming_normal_(self.kernel_widths, mode="fan_in", nonlinearity="relu")
    if self.sqrt_precision is not None:
      nn.init.kaiming_normal_(self.sqrt_precision, mode="fan_in", nonlinearity="relu")

  """r
    x: (N,L,kmax,n_coord), where n_coord = d, is coordinate dimension
  """
  def forward(self, x):
    nbatch_dim   = len(x.shape) - 1
    input_size   = torch.Size([1 for _ in range(nbatch_dim)]) # (1,1,1)
    centers_size = input_size + self.center_shape # (1,1,1,n_coord,N)

    if self.covariance_type == 'diag':
      base_size = input_size + self.width_shape # [1,1,1,5,N]
      # (bs,seq,32,5,1) - (1,1,1,d=5,N) = (bs,seq,32,5,N)
      base = (kernel_widths + self.eps).reshape(base_size)
      x = ( x.unsqueeze(dim=-1) - self.kernel_centers.reshape(centers_size) ) / base
      activity = torch.exp( -0.5 * torch.sum(x**2, dim=-2) )
    elif self.covariance_type == 'full':
      # (bs,seq,kmax,n_coord,1) - (1,1,1,n_coord,N) = (bs,seq,kmax,n_coord,N)
      # (bs,seq,16,d=3,1) - (1,1,1,d=3,N) = (bs,seq,16,d-3,N)
      intermediate  = x.unsqueeze(dim=-1) - self.kernel_centers.reshape(centers_size)
      # (bs,seq,16,1,d,N) * (1,d,d,N) = (bs,seq,16,d,d,N) = (bs,seq,16,d,N)
      intermediate2 = torch.sum(intermediate.unsqueeze(dim=-3) 
                                * self.sqrt_precision.unsqueeze(dim=0), dim=-2)
      activity = torch.exp(-0.5 * torch.sum(intermediate2**2, dim=-2)) # (bs,seq,32,N)
    else:
      activity = None
    return activity


# G(x) * a + G(x) + bias
class EmbeddingOuterProduct(nn.Module):
  def __init__(self, config):
    super(EmbeddingOuterProduct, self).__init__()
    self.config = config # not used any more

    self.sum_axis = 2
    self.use_bias = False
    self.kernel12 = torch.nn.Parameter(data=torch.Tensor(32, 20, 128), requires_grad=True)
    self.kernel1 = torch.nn.Parameter(data=torch.Tensor(32, 128), requires_grad=True)
    self.bias = torch.nn.Parameter(data=torch.Tensor(128), requires_grad=True)
    # init

  def forward(self, inputs):
    first_input = inputs[0] # [bs, seq, k, 32]
    second_input = inputs[1] # [bs, seq, k, 20]

    if self.sum_axis is not None:
      temp = torch.unsqueeze(first_input, dim=-1) \
            * torch.unsqueeze(second_input, dim=-2) # (bs,seq,k,32,1)*(bs,seq,k,1,20)
      outer_product = torch.sum(temp, dim=self.sum_axis) # (bs,seq,32,20) sum across dim kmax

    activity = torch.tensordot(outer_product, self.kernel12, dims=([-2, -1], [0, 1])) #(bs,sq,128)

    activity += torch.tensordot(first_input.sum(dim=self.sum_axis), self.kernel1, ([-1],[0]))
    if self.use_bias:
      activity += self.bias.reshape(1, 1, -1)
    return activity

