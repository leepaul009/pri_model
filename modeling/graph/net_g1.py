


import torch
from torch import Tensor, nn
from torch.nn import functional as F


from modeling.graph.layers import Linear
from modeling.graph.neighborhoods import LocalNeighborhood
from modeling.graph.embeddings import GaussianKernel, EmbeddingOuterProduct


class GNetVerOne(nn.Module):

  def __init__(self, hidden_dim=32, ):
    super(GNetVerOne, self).__init__()

    config = None
    coordinates = ['euclidian', 'index_distance', 'ZdotZ', 'ZdotDelta']

    self.local_neighborhood = LocalNeighborhood(config, Kmax=16, 
                                                coordinates=coordinates, 
                                                self_neighborhood=True, 
                                                index_distance_max=8, 
                                                nrotations=1)
    d_gaussians = 3+1+1+1+1
    n_gaussians = 32
    initial_values = ( torch.zeros(3,n_gaussians).float(), torch.zeros(3,3,n_gaussians).float() )
    covariance_type = "full"
    self.guassian_kernel = GaussianKernel(d=d_gaussians,
                                          N=n_gaussians, 
                                          initial_values=initial_values, 
                                          covariance_type=covariance_type)

    self.outer_product = EmbeddingOuterProduct(config=None)


  def forward(self, x):
    # (bs, L, 20) => (bs, L, 32)

    # 第一个输入 frame，第二个输入 index
    # first element of input is "frame"
    # second element of input is "index"
    # aa_attributes: (N,L,20)
    # frame:         (N,L,4,3)
    # aa_indices:    (N,L,1)
    # labels:        (N,)
    aa_attributes, frame, aa_indices, labels = x

    # outputs: 
    # neighbor_coords: List[Tensor]: 
    #   euclidian:  (N,L,kmax,3)
    #   ZdotZ:      (N,L,kmax,1)
    #   DeltadotZ:  (N,L,kmax,1)
    #   ZdotDelta:  (N,L,kmax,1)
    #   index_dist: (N,L,kmax,1)
    # neighbor_attr: (N, L, kmax, H)
    neighbor_coords, neighbor_attr = self.local_neighborhood( [frame, aa_indices, aa_attributes] )

    # euclidian_coords = neighbor_coords[0] # (N,L,kmax,3)
    neighbor_coords_feat = torch.cat(neighbor_coords, dim=-1)

    # (N,L,kmax,3) ==> (N,L,kmax,32)
    embedded_local_coordinates = self.guassian_kernel(neighbor_coords_feat)

    # (N,L,kmax,32), (N,L,kmax,H) ==> (N,L,128) where H(outer_product) = 128
    filters_input = self.outer_product( [embedded_local_coordinates, neighbor_attr] )




    return None