import numpy as np
import os
import sys


import torch
from torch import Tensor, nn
from torch.nn import functional as F

def gather_nd_torch(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    # indice_dims = indices.size()[batch_dim:-1]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)
    # indices = indices.reshape(batch_size, *indice_dims, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    # batch_pad_dim = torch.Size([1 for _ in range(len(indice_dims))])
    # batch_enumeration = torch.arange(batch_size).reshape(-1, *batch_pad_dim)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    # gather_dims = [indices[..., i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def gather_nd_torch_ex(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x1, ..., xQ, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x1, ..., xQ, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    indice_dims = indices.size()[batch_dim:-1] # [x1, .. xQ]
    # n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    # indices = indices.reshape(batch_size, n_indices, n_pos)
    indices = indices.reshape(batch_size, *indice_dims, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    # batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    batch_pad_dim = torch.Size([1 for _ in range(len(indice_dims))])
    batch_enumeration = torch.arange(batch_size).reshape(-1, *batch_pad_dim) # [bs, 1<1>,... 1<Q>]
    # gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims = [indices[..., i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    # gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    gathered = gathered.reshape(*batch_dims, *indice_dims, c_dim)
    return gathered



class FrameBuilder(nn.Module):
  def __init__(self, config, order='1', dipole=False):
    super(FrameBuilder, self).__init__()
    self.support_masking = True
    self.epsilon = torch.tensor(1e-6)
    self.order = order # atom frame use 2
    self.dipole = dipole # false

    self.xaxis = torch.tensor(np.array([[1, 0, 0]], dtype=np.float32))
    self.yaxis = torch.tensor(np.array([[0, 1, 0]], dtype=np.float32))
    self.zaxis = torch.tensor(np.array([[0, 0, 1]], dtype=np.float32))
  """
  input:
    inputs: 
      points Tensor=(N, all_atoms, 3), 
      triplets(atom_coord:x/y/z) Tensor=(N, all_hatoms?, 3), each item is point index
    mask: Tensor=(N, all_atoms) Tensor=(N, all_hatoms?)
  """
  def forward(self, inputs, mask=None):
    points, triplets = inputs
    # clip value of each item of triplets in range of 0~points.shape[-2],
    # as each item of triplets indicate which point(in points) the item belong
    triplets = torch.clamp(triplets, min=0, max=points.shape[-2]-1)

    # triplets[:, :, 1:2].shape = (?,?,1)
    # gather_nd_torch(points, triplets[:, :, 0:1], batch_dim=1)的含义：points的shape是(batch, all_atoms, 3)，triplets的shape是(batch, ?, 3)，
    # triplets[:,:,0:1]表示的是三元组中的第一个元素，如果三元组表示为（中，前，后），则其表示为中间的元素，其值是atom的序号；
    # gather_nd_torch意为从points的第二维(atoms)中选取triplets第一个元素记录的atom序号，从points中找到对应序号的atom的三个坐标值
    # delta_10可以理解为向量，从center指向front，delta_20是center指向rear的向量
    delta_10 = gather_nd_torch(points, triplets[:, :, 1:2], batch_dim=1) - gather_nd_torch(points, triplets[:, :, 0:1], batch_dim=1)
    delta_20 = gather_nd_torch(points, triplets[:, :, 2:3], batch_dim=1) - gather_nd_torch(points, triplets[:, :, 0:1], batch_dim=1)

    if self.order in ['2','3']: 
      delta_10,delta_20 = delta_20,delta_10
    
    # 中心点很好理解
    centers = gather_nd_torch(points, triplets[:, :, 0:1], batch_dim=1)
    # z轴是：向量center->front的单位向量
    zaxis = (delta_10 + self.epsilon * torch.reshape(self.zaxis,[1,1,3])) / (torch.sqrt(torch.sum(delta_10 ** 2, dim=-1, keepdim=True) ) + self.epsilon)

    # y轴是：z轴和向量center->rear组成的平面的垂直向量
    yaxis = torch.cross(zaxis, delta_20, dim=-1)
    yaxis = (yaxis + self.epsilon * torch.reshape(self.yaxis,[1,1,3]) ) / (torch.sqrt(torch.sum(yaxis ** 2, dim=-1, keepdim=True) ) + self.epsilon)

    xaxis = torch.cross(yaxis, zaxis, dim=-1)
    xaxis = (xaxis + self.epsilon * torch.reshape(self.xaxis,[1,1,3]) ) / (torch.sqrt(torch.sum(xaxis ** 2, dim=-1, keepdim=True)) + self.epsilon)

    # TODO: check if following code is necessary
    if self.order == '3':
      xaxis,yaxis,zaxis = zaxis,xaxis,yaxis

    if self.dipole:
      # TODO: check if following code is necessary
      dipole = gather_nd_torch(points, triplets[:, :, 3:4], batch_dim=1) - gather_nd_torch(points, triplets[:, :, 0:1],batch_dim=1)
      dipole = (dipole + self.epsilon * torch.reshape(self.zaxis, [1,1,3])) / (torch.sqrt(torch.sum(dipole ** 2, dim=-1, keepdim=True) ) + self.epsilon)
      frames = torch.stack([centers,xaxis,yaxis,zaxis,dipole],dim=-2)
    else:
      frames = torch.stack([centers,xaxis,yaxis,zaxis],dim=-2) # (batch, n_aa/atom[stack_dim], 3)

    if mask not in [None,[None,None]]:
      frames *= mask[-1].type(torch.float32).unsqueeze(dim=-1).unsqueeze(dim=-1)
    return frames


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

    if self.nrotations > 1:
      phis = np.arange(self.nrotations) / self.nrotations * 2 * np.pi
      rotations = np.zeros([self.nrotations, 3, 3], dtype=np.float32)
      # rotatioin along z-aixs
      rotations[:, 0, 0] = np.cos(phis)
      rotations[:, 1, 1] = np.cos(phis)
      rotations[:, 1, 0] = np.sin(phis)
      rotations[:, 0, 1] = -np.sin(phis)
      rotations[:, 2, 2] = 1
      # self.rotations = torch.tensor(rotations)
      self.rotations = rotations


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

    # if 'frame' in self.second_format:
    #   batch_size = first_frame.shape[0]
    # if 'index' in self.first_format:
    #   assert first_index.shape[0] == second_index.shape[0]
    #   batch_size = first_index.shape[0]
    batch_size = inputs[0].shape[0]

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
    
    if 'euclidian' in self.coordinates:
      nei = neighbors.unsqueeze(dim=-1)
      # (bs, s1, k, 3) - (bs, s1, 1, 3)
      temp = gather_nd_torch_ex(second_center, nei, batch_dim=1) - first_center.unsqueeze(dim=-2)
      # (bs, s1, k, 1, 3)
      temp = temp.unsqueeze(dim=-2)
      # (bs, s1, 1, 4->3, 3) => (bs, s1, k, 3, 3)
      temp = temp * first_frame[:,:,1:4].unsqueeze(dim=-3)
      euclidian_coordinates = torch.sum(temp, dim=-1) # (bs, s1, k, 3)

      if self.nrotations > 1:
        # rotations: Tensor=(nrotations, 3, 3)
        rotations = torch.tensor(self.rotations).to(device)
        # dot product with last dim == matmul: roation_mat(3,3).vec(3)
        # |0 -1 0| |1| |0|
        # |1  0 0|.|0|=|1|
        # |0  0 1| |0| |0|， when phi=90 or pi/2, equal to dot product with rmat's last dim
        # (bs, s1, k, 3) dot (nr, 3, 3) => (bs, s1, k, nr, 3)
        euclidian_coordinates = torch.tensordot(euclidian_coordinates, rotations, dims=([-1],[-1]))
        neighbors_attributes = [nei_att.unsqueeze(dim=-2) for nei_att in neighbors_attributes]

      neighbor_coordinates.append(euclidian_coordinates)

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
