import numpy as np
import os
import sys


import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrameBuilder(nn.Module):
  def __init__(self, config):
    super(FrameBuilder, self).__init__()


  def forward(self, inputs):

    return inputs


# coordinates1/coordinates2: [N, seq, hidden]
def distance(coordinates1,coordinates2,squared=False,ndims=3):
  # [N, seq, 1] - [N, 1, seq]
  D = ( coordinates1[...,0].unsqueeze(dim=-1) - coordinates2[...,0].unsqueeze(dim=-2) )**2
  for n in range(1,ndims):
    D += ( coordinates1[...,n].unsqueeze(dim=-1) - coordinates2[...,n].unsqueeze(dim=-2) )**2
  if not squared:
    D = torch.sqrt(D)
  return D


class LocalNeighborhood(nn.Module):
  def __init__(self, 
      config,
      Kmax=10, 
      coordinates=['index_distance'],
      self_neighborhood=True,
      index_distance_max=None,
      nrotations = 1):
    super(LocalNeighborhood, self).__init__()
    self.config = config

    self.Kmax = Kmax
    self.coordinates = coordinates
    self.self_neighborhood = self_neighborhood

    self.first_format = []
    self.second_format = []
    if (('euclidian' in self.coordinates) 
        | ('ZdotZ' in self.coordinates) 
        | ('ZdotDelta' in self.coordinates)):
      self.first_format.append('frame')
      if (self.self_neighborhood 
          | ('ZdotZ' in self.coordinates) 
          | ('ZdotDelta' in self.coordinates)):
        self.second_format.append('frame')
      else:
        # cross neighbor and euclidian
        self.second_format.append('point')
    elif 'distance' in self.coordinates:
      self.first_format.append('point')
      self.second_format.append('point')
    
    if 'index_distance' in self.coordinates:
      self.first_format.append('index')
      self.second_format.append('index')

    self.index_distance_max = index_distance_max
    self.epsilon = 1e-10
    self.big_distance = 1000.
    self.nrotations = nrotations


  # inputs: indices1[N,s1,1], indices2[N,s2,1], attr1[N,s1,h], attr2[N,s2,h]
  # outputs:
  def forward(self, inputs):
    
    # TODO: check input[0] is a tensor
    device = inputs[0].device

    # get frame data
    if 'frame' in self.first_format:
      first_frame = inputs[self.first_format.index('frame')]
    else:
      first_frame = None
    if 'frame' in self.second_format:
      if self.self_neighborhood:
        second_frame = first_frame
      else:
        second_frame = inputs[len(self.first_format) + self.second_format.index('frame')]
    else:
      second_frame = None

    # get point data
    if 'point' in self.first_format:
      first_point = inputs[self.first_format.index('point')]
    else:
      first_point = None
    if 'point' in self.second_format:
      if self.self_neighborhood:
        second_point = first_point
      else:
        second_point = inputs[len(self.first_format) + self.second_format.index('point')]
    else:
      second_point = None

    # get index data
    if 'index' in self.first_format:
      first_index = inputs[self.first_format.index('index')] # [bs, s1, 1]
    else:
      first_index = None
    if 'index' in self.second_format:
      if self.self_neighborhood:
        second_index = first_index
      else:
        second_index = inputs[len(self.first_format)+self.second_format.index('index')] # [bs, s2, 1]
    else:
      second_index = None


    assert first_index.shape[0] == second_index.shape[0]
    batch_size = first_index.shape[0]

    # prepare input feature
    # list( [bs, s2, h] )
    nattributes = len(inputs) - len(self.first_format) - (1-1*self.self_neighborhood) * len(self.second_format)
    # print("nattributes={}".format(nattributes))
    second_attributes = inputs[-nattributes:]

    # determine the first and second center
    if first_frame is not None:
      first_center = first_frame[:,:,0]
      ndims = 3
    elif first_point is not None:
      first_center = first_point
      ndims = 3
    else:
      first_center = first_index.type(torch.float32)
      ndims = 1

    if second_frame is not None:
      second_center = second_frame[:,:,0]
    elif second_point is not None:
      second_center = second_point
    else:
      second_center = first_index.type(torch.float32)

    # [bs, s1, s2]
    distance_square = distance(first_center, second_center, squared=True, ndims=ndims)

    # neighbors = torch.unsqueeze(torch.argsort(distance_square)[:,:,:self.Kmax], dim=-1)
    neighbors = torch.argsort(distance_square, dim=-1)[:, :, :self.Kmax] # [bs, s1, k]
    # neighbors = neighbors.view(-1, self.Kmax) # TODO: continous?? [bs*s1, k]


    neighbors_attributes = []
    for attribute in second_attributes:
      assert attribute.shape[0] == batch_size # [bs, s2, h2]
      attr_per_batch = []
      for i in range(batch_size):
        # attribute[i]=(s2, h2) neighbors[i]=(s1, k)
        neighbors_attr = attribute[i][neighbors[i], :] # [s1, k, h2]
        attr_per_batch.append(neighbors_attr.unsqueeze(dim=0)) # [1, s1, k, h2]
      attr_per_batch = torch.cat(attr_per_batch, dim=0) # [bs, s1, k, h2]
      neighbors_attributes.append(attr_per_batch)
    # print("neighbors_attributes size={}".format(len(neighbors_attributes)))

    self.epsilon = torch.tensor(self.epsilon).to(device)
    neighbor_coordinates = []
    
    # if 'euclidian' in self.coordinates:
    if 'distance' in self.coordinates:
      # distance_square=(bs, s1, s2), neighbors=(bs, s1, k)
      bs, s1, s2 = distance_square.shape
      offset = torch.arange(bs * s1).reshape(bs, s1, 1) * s2
      nei_offset = offset.to(device) + neighbors # [bs,s1,1]+[bs,s1,k]=[bs,s1,k]
      # match k neighbors across bs*s1 dim
      distance_neighbors = distance_square.reshape(-1)[nei_offset.reshape(-1)]
      distance_neighbors = distance_neighbors.reshape(bs, s1, self.Kmax, 1)
      neighbor_coordinates.append(distance_neighbors)

    if 'ZdotZ' in self.coordinates:
      # frame=[bs,s1,4,3]
      first_zdirection = first_frame[:,:,-1] # [bs,s1,3]
      second_zdirection = second_frame[:, :, -1] # [bs,s2,3]
      # neighbors=(bs, s1, k)
      bs, s1, h1 = first_zdirection.shape
      _, s2, h2 = second_zdirection.shape
      
      offset = torch.arange(bs).reshape(-1, 1, 1) * s2
      nei_offset = offset.to(device) + neighbors
      
      nei_second_zdir = second_zdirection.reshape(bs*s2, -1)[nei_offset.reshpae(-1)]
      nei_second_zdir = nei_second_zdir.reshape(bs, s1, self.Kmax, -1)
      # vector dot product
      ZdotZ_neighbors = first_zdirection.unsqueeze(dim=-2) * nei_second_zdir
      ZdotZ_neighbors = torch.sum(ZdotZ_neighbors, dim=-1, keepdim=True) # [bs, s1, k, 1]

      neighbor_coordinates.append(ZdotZ_neighbors)

    if 'ZdotDelta' in self.coordinates:
      first_zdirection = first_frame[:,:,-1]
      second_zdirection = second_frame[:, :, -1]
      bs, s1, h1 = first_zdirection.shape
      _, s2, h2 = second_zdirection.shape

      # TODO: same to ZdotZ, use single
      offset = torch.arange(bs).reshape(-1, 1, 1) * s2
      nei_offset = offset.to(device) + neighbors

      nei_center = second_center.reshape(bs*s2, -1)[nei_offset.reshpae(-1)]
      nei_center = nei_center.reshape(bs, s1, self.Kmax, -1)
      # TODO: self.epsilon is float scalar
      # center_vector from first center to second neighbor center
      DeltaCenter_neighbors = (
        nei_center - first_center.unsqueeze(dim=-2)) / (distance_neighbors + self.epsilon)
      # z_vector dot product with center_vector
      ZdotDelta_neighbors = torch.sum(first_zdirection.unsqueeze(dim=-2) * DeltaCenter_neighbors,
        dim=-1, keepdim=True) # [bs, s1, k, 3] => [bs, s1, k](dot product sum x/y/z)
      # TODO: depends on 'ZdotZ' computation
      DeltadotZ_neighbors = torch.sum(DeltaCenter_neighbors * nei_second_zdir, dim=-1, keepdim=True)

      neighbor_coordinates.append(DeltadotZ_neighbors)
      neighbor_coordinates.append(ZdotDelta_neighbors)


    if 'index_distance' in self.coordinates:
      neighbor_second_indices = []
      for i in range(batch_size):
        # [s2, 1] => [s1, k, 1] => [1, s1, k, 1]
        neighbor_second_indices.append(
          second_index[i, neighbors[i], :].unsqueeze(dim=0) )
      neighbor_second_indices = torch.cat(neighbor_second_indices, dim=0) # [bs, s1, k, 1]

      index_distance = first_index.unsqueeze(dim=-2) - neighbor_second_indices
      index_distance = torch.abs(index_distance.type(torch.float32))

      neighbor_coordinates.append(index_distance)
    # print("neighbor_coordinates size={}".format(len(neighbor_coordinates)))

    output = [neighbor_coordinates] + neighbors_attributes
    return output