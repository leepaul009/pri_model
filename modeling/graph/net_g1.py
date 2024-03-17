


import torch
from torch import Tensor, nn
from torch.nn import functional as F


from modeling.graph.layers import Linear
from modeling.graph.neighborhoods import LocalNeighborhood
from modeling.graph.embeddings import GaussianKernel, \
  EmbeddingOuterProduct, AttributeEmbedding, AttentionLayer



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

    self.outer_product = EmbeddingOuterProduct(n_gaussians=n_gaussians,  #32
                                               in_features=20,
                                               out_features=128)

    self.relu = nn.ReLU(inplace=True)
    
    self.attribute_embedding = AttributeEmbedding(in_features=128, 
                                                  out_features=32, 
                                                  activation='relu')

    self.beta = AttributeEmbedding(in_features=32, out_features=1, activation='relu')
    self.self_attention = AttributeEmbedding(in_features=32, out_features=1, activation='relu')
    self.cross_attention = AttributeEmbedding(in_features=32, out_features=1, activation='relu')
    self.node_features = AttributeEmbedding(in_features=32, out_features=2, activation='relu')

    # TODO: 确保distance能用
    coordinates = ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance']
    self.att_neighborhood = LocalNeighborhood(config, Kmax=32, 
                                              coordinates=coordinates, 
                                              self_neighborhood=True, 
                                              index_distance_max=8, 
                                              nrotations=1)
    d_gaussians = 5 # 网络输入的 hidden维度
    self.guassian_kernel2 = GaussianKernel(d=d_gaussians,
                                          N=n_gaussians, 
                                          initial_values=initial_values, 
                                          covariance_type=covariance_type)
    self.attribute_embedding2 = AttributeEmbedding(in_features=32, 
                                                  out_features=1, 
                                                  activation='relu')
    self.attention_layer = AttentionLayer()

  def forward(self, x):
    # (bs, L, 20) => (bs, L, 32)

    # 第一个输入 frame，第二个输入 index
    # first element of input is "frame"
    # second element of input is "index"
    # aa_attributes: (N,L,20)
    # frame:         (N,L,4,3)
    # aa_indices:    (N,L,1)
    # labels:        (N,)
    aa_attributes, aa_frame, aa_indices, labels = x

    """
      outputs: 
        neighbor_coords: List[Tensor]: 
          euclidian:  (N,L,kmax,3)
          ZdotZ:      (N,L,kmax,1)
          DeltadotZ:  (N,L,kmax,1)
          ZdotDelta:  (N,L,kmax,1)
          index_dist: (N,L,kmax,1)
        neighbor_attr: (N, L, kmax, H=20), H维度是aa_attributes的维度, H=20
    """
    neighbor_coords, neighbor_attr = self.local_neighborhood( [aa_frame, aa_indices, aa_attributes] )

    # euclidian_coords = neighbor_coords[0] # (N,L,kmax,3)
    neighbor_coords_feat = torch.cat(neighbor_coords, dim=-1)

    # (N,L,kmax,3) ==> (N,L,kmax,32)
    embedded_local_coordinates = self.guassian_kernel(neighbor_coords_feat)

    # (N,L,kmax,32), (N,L,kmax,H) ==> (N,L,128) where H(outer_product) = 128
    filters_input = self.outer_product( [embedded_local_coordinates, neighbor_attr] )

    SCAN_filters_aa = self.relu(filters_input)
    # do dropout to SCAN_filters_aa

    # (N,L,128) => (N,L,32)
    embedded_filter = self.attribute_embedding(SCAN_filters_aa)

    beta = self.beta(embedded_filter) # (N,L,1)
    self_attention = self.self_attention(embedded_filter) # (N,L,1)
    cross_attention = self.cross_attention(embedded_filter)

    node_features = self.node_features(embedded_filter)

    # TODO: aa_indices确保能在这个net里用，不会有bug
    # 这里用了self_attention=True
    # 网络的first_format/second_format = [frame, index]
    # 前两个输入 aa_frame, aa_indices 分别被用于att_neighborhood的 frame，index 输入
    # 后两个输入 cross_attention, node_features 作为特征，用于作为邻居特征输出

    # 输出：List((bs,seq,k=32,1)), (bs,seq,k=32,1), (bs,seq,k=32,2)
    graph_weights, attention_local, node_features_local =\
      self.att_neighborhood( [aa_frame, aa_indices, cross_attention, node_features] )
    graph_weights = torch.cat(graph_weights, dim=-1) # (bs,seq,k=32,5)

    # (bs,seq,k=32,5) = (bs,seq,k=32,32)
    embedded_graph_weights = self.guassian_kernel2(graph_weights) 
    # (bs, seq, 32, 1)
    embedded_graph_weights = self.attribute_embedding2(embedded_graph_weights)

    # (N,L,1) (N,L,1) (N,L,32,1), (N,L,32,2) (N,L,32,1) = (bs, seq, 2) (bs, seq, 32, 1)
    graph_attention_output, attention_coefficients = self.attention_layer(
      [beta, self_attention, attention_local, 
      node_features_local, embedded_graph_weights])


    # output: graph_attention_output



    return None