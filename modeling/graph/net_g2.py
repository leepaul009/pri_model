import torch
from torch import Tensor, nn
from torch.nn import functional as F


from modeling.graph.layers import Linear
from modeling.graph.neighborhoods import LocalNeighborhood
from modeling.graph.embeddings import GaussianKernel, \
  EmbeddingOuterProduct, AttributeEmbedding, AttentionLayer
from modeling.lib import CrossAttention


class GNetVerSecond(nn.Module):

  def __init__(self, hidden_dim=32, ):
    super(GNetVerSecond, self).__init__()

    config = None
    coordinates = ['euclidian', 'index_distance', 'ZdotZ', 'ZdotDelta']
    self.local_neighborhood = LocalNeighborhood(config, Kmax=16, 
                                                coordinates=coordinates, 
                                                self_neighborhood=True, 
                                                index_distance_max=8, 
                                                nrotations=1)
    
    d_gaussians = 3+1+1+1+1
    n_gaussians = 32
    self.n_gaussians = n_gaussians
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

    # hidden_size = attention_head_size * num_attention_heads
    num_attention_heads = 8
    attention_head_size = 32 # 每个head的特征维度
    hidden_size = attention_head_size * num_attention_heads # 128 网络内部参数

    key_hidden_size = 20
    query_hidden_size = 3+1+1+1+1

    self.cross_att = CrossAttention(hidden_size = hidden_size, 
                                    attention_head_size = attention_head_size,
                                    num_attention_heads = num_attention_heads,
                                    key_hidden_size = key_hidden_size,
                                    query_hidden_size = query_hidden_size)

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
        neighbor_attr: (N, L, kmax, H=20), H维度是aa_attributes的维度, H=20(氨基酸种类)
    """
    ###### 1) 计算每个node想对于k近邻node的几何特征，即neighbor_coords
    neighbor_coords, neighbor_attr = self.local_neighborhood( [aa_frame, aa_indices, aa_attributes] )

    # neighbor_coords_feat是每个node的k近邻的某种特征，即每个氨基酸的k近邻的氨基酸的特征：
    #   距离(node的z轴和k近邻node的距离)
    #   角度(node的z轴和k近邻node的z轴夹角)等
    #   注意：每个“邻居氨基酸”的特征维度=7, 代表了空间属性
    # euclidian_coords = neighbor_coords[0] # (N,L,kmax,3)  ## 只是用距离作为“邻居氨基酸”的特征
    neighbor_coords_feat = torch.cat(neighbor_coords, dim=-1) ## 选用所有作为“邻居氨基酸”的特征

    ###### 2)
    # neighbor_coords_feat (N,L,kmax,7)
    # neighbor_attr (N,L,kmax,20)
    batch_size = neighbor_coords_feat.shape[0]
    seq_len = neighbor_coords_feat.shape[1]
    kmax = neighbor_coords_feat.shape[2]
    query_feat = neighbor_coords_feat.reshape(batch_size * seq_len, kmax, -1)  # （N*L, k, 7）
    feat = neighbor_attr.reshape(batch_size * seq_len, kmax, -1)  # (N*L, k, 20)

    y = self.cross_att(hidden_states_query = query_feat, hidden_states_key = feat)

     
    # y = e**[ -0.5*((x - ctr_kernel) * sqrt_kernel)**2 ]
    # (N,L,kmax,7) ==> (N,L,kmax,32)
    embedded_local_coordinates = self.guassian_kernel(neighbor_coords_feat)
    # (N,L,kmax,32), (N,L,kmax,H) ==> (N,L,128) where H(outer_product) = 128
    filters_input = self.outer_product( [embedded_local_coordinates, neighbor_attr] )
    SCAN_filters_aa = self.relu(filters_input)
    # do dropout to SCAN_filters_aa

    # (N,L,128) => (N,L,32)
    embedded_filter = self.attribute_embedding(SCAN_filters_aa)

    beta = self.beta(embedded_filter) # (N,L,1)
    self_attention = self.self_attention(embedded_filter) # (N,L,1)
    cross_attention = self.cross_attention(embedded_filter) # (N,L,1)

    node_features = self.node_features(embedded_filter) # (N,L,2)

    # TODO: aa_indices确保能在这个net里用，不会有bug
    # 这里用了self_attention=True
    # 网络的first_format/second_format = [frame, index]
    # 前两个输入 aa_frame, aa_indices 分别被用于att_neighborhood的 frame，index 输入
    # 后两个输入 cross_attention, node_features 作为特征，用于作为邻居特征输出

    # 输出：List((bs,seq,k=32,1)), attention_local: (bs,seq,k=32,1), node_features_local: (bs,seq,k=32,2)
    graph_weights, attention_local, node_features_local =\
      self.att_neighborhood( [aa_frame, aa_indices, cross_attention, node_features] )
    graph_weights = torch.cat(graph_weights, dim=-1) # (bs,seq,k=32,5)

    # (bs,seq,k=32,5) = (bs,seq,k=32,32)
    embedded_graph_weights = self.guassian_kernel2(graph_weights) 
    # (bs, seq, 32, 1) -> (bs*seq, 32, 1)
    tmp_dim_2, tmp_dim_3 = embedded_graph_weights.shape[2], embedded_graph_weights.shape[2]
    embedded_graph_weights = embedded_graph_weights.reshape(batch_size*seq_len, tmp_dim_2, tmp_dim_3)
    embedded_graph_weights = self.attribute_embedding2(embedded_graph_weights)
    embedded_graph_weights = embedded_graph_weights.reshape(batch_size, seq_len, tmp_dim_2, -1)

    # (N,L,1) (N,L,1) (N,L,32,1), (N,L,32,2) (N,L,32,1) = (bs, seq, 2) (bs, seq, 32, 1)
    graph_attention_output, attention_coefficients = self.attention_layer(
      [beta, self_attention, attention_local, 
      node_features_local, embedded_graph_weights])


    # output: graph_attention_output

    # graph_attention_output：是float, attention_coefficients: 都是1,0,用于分类
    # graph_attention_output: (bs, seq, 2), attention_coefficients: (bs, seq, 32, 1)
    return graph_attention_output, attention_coefficients