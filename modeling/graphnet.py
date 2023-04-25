from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed
import utils


class SubGraph(nn.Module):
    # config:
    #   depth:int, hidden_size:int, point_level-4-3:bool
    def __init__(self, config, hidden_size, depth):
        super(SubGraph, self).__init__()
        
        # depth = config['depth']
        # hidden_size = config['hidden_size']

        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) 
                                     for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        
        if config['point_level-4-3']:
            self.layer_0_again = MLP(hidden_size)
        
        if config['use_atom_embedding']:
            atom_nfeatures = config['atom_nfeatures']
            self.atom_emb_layer = nn.Embedding(num_embeddings=atom_nfeatures + 1, 
                                               embedding_dim=atom_nfeatures)
            self.layer_0 = MLP(atom_nfeatures, hidden_size)
        
        self.config = config


    # input_list:
    #   case1: a chain, shape=residues*[atoms, hidden] (att along atoms)
    #   case2: a batch of chains, shape=N*[residues, hidden] (att along residues)
    def forward(self, input_list : List[Tensor]):
        batch_size = len(input_list)
        device = input_list[0].device
        # input_list is list of tensor[nodes(atom/aa), hidden]
        # merge with padding: hidden_states = [lines, max_nodes, hidden]
        hidden_states, lengths = utils.merge_tensors(input_list, device) # [aa, max_atoms, h]
        # hidden_size = hidden_states.shape[2] # here hidden size is input's
        max_vector_num = hidden_states.shape[1] # do att along this dim

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device)
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            hidden_states = self.atom_emb_layer(hidden_states)

        hidden_states = self.layer_0(hidden_states)

        if self.config['point_level-4-3']:
            hidden_states = self.layer_0_again(hidden_states)
        
        # mask sequence dim as each item in batch has difference seq length
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        # apply max pooling along seq dim, output: [N, h]
        return torch.max(hidden_states, dim=1)[0], \
               torch.cat(utils.de_merge_tensors(hidden_states, lengths))



class GraphNet(nn.Module):
    r"""
    It has two main components, sub graph and global graph.
    Sub graph encodes a polyline as a single vector.

    n_aa: number of aa
    n_atoms: number of atoms in one aa
    """

    def __init__(self, config, args_: utils.Args):
        super(GraphNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.hidden_size = hidden_size
        
        # depth:int, hidden_size:int, point_level-4-3:bool
        atom_nfeatures = 12
        self.atom_emb_layer = nn.Embedding(num_embeddings=atom_nfeatures + 1, 
                                           embedding_dim=args.hidden_size)
        
        # config['depth'] = args.sub_graph_depth
        # config['hidden_size'] = args.hidden_size
        config['point_level-4-3'] = True
        config['use_atom_embedding'] = True
        config['atom_nfeatures'] = 12
        self.point_level_sub_graph = SubGraph(config, hidden_size // 2, args.sub_graph_depth)
        # self.point_level_cross_attention = CrossAttention(hidden_size)

        # self.global_graph = GlobalGraph(hidden_size)
        # output dim = hidden_size
        self.global_graph = GlobalGraphRes(hidden_size)
        
        self.laneGCN_A2L = CrossAttention(hidden_size // 2)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size // 2)
        self.laneGCN_L2A = CrossAttention(hidden_size // 2)

        # TODO: adjust hidden size to 32 (used in ScanNet)
        # TODO: make 20 as param
        # TODO: make 2 as param
        # TODO: make 'relu' as para
        self.aa_embeding_layer = nn.RNN(
            input_size=20, hidden_size=hidden_size // 2, num_layers=2, 
            nonlinearity='relu', batch_first=True)
        
        self.aa_embeding_layer = nn.Linear(in_features=20, out_features=hidden_size // 2, bias=False)
        self.aa_embeding_layer = TimeDistributed(self.aa_embeding_layer)

        # TODO: if we 
        self.reg_pred = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        # TODO: init weight and bias
        # nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.001)
        for l in [self.reg_pred]:
            nn.init.constant_(l.bias, 0)

    def forward_encode_sub_graph(
            self, 
            mapping: List[Dict], 
            atom_attributes, # List[List[np.ndarray]]
            atom_indices,
            # matrix: List[np.ndarray], 
            # polyline_spans: List[List[slice]],
            device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """

        """
        input_list_list = []
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)
        """
        
        prot_atom_input_list_list = []
        drna_atom_input_list_list = []

        for i in range(batch_size): # per chain
            input_list = []
            for j in range(len(atom_attributes[i])): # per aa
                # np.array to tensor shape=[num_atoms, 1]
                tensor = torch.tensor(atom_attributes[i][j], device=device)
                input_list.append(tensor)
            prot_atom_input_list_list.append(input_list)

        prot_states_batch = []
        for i in range(batch_size): # per chain
            a, b = self.point_level_sub_graph(prot_atom_input_list_list[i])
            prot_states_batch.append(a)

        rdna_states_batch = []
        for i in range(len(drna_atom_input_list_list)): # per chain
            a, b = self.point_level_sub_graph(drna_atom_input_list_list[i])
            rdna_states_batch.append(a)

        for i in range(batch_size):
            prot_feats = prot_states_batch[i] # [aa, h/2]
            rdna_feats = rdna_states_batch[i] # [nc, h/2]
            rdna_feats = rdna_feats + self.laneGCN_A2L(
                rdna_feats.unsqueeze(0), prot_feats.unsqueeze(0)).squeeze(0)
            rdna_feats = rdna_feats + self.laneGCN_L2L(rdna_feats.unsqueeze(0)).squeeze(0)
            prot_feats = prot_feats + self.laneGCN_L2A(
                prot_feats.unsqueeze(0), rdna_feats.unsqueeze(0)).squeeze(0)
            # prot_states_batch[i] = torch.cat([prot_feats, rdna_feats]) # [aa+nc, h]
            prot_states_batch[i] = prot_feats # [aa, h]

        return prot_states_batch, rdna_states_batch
    
    def attribute_embeding(
            self,
            aa_attributes : List[np.ndarray], 
            # aa_indices,
            device, batch_size) -> Tuple[Tensor, List[int]]:
        """
        inputs:
            aa_attributes: List[Tensor=(n_aa, 20)]
        outputs:
            List[Tensor=(n_aa, h)]
            Tensor=(bs, max(n_aa), h)
        """
        input_list = []
        for i in range(batch_size):
            tensor = torch.tensor(aa_attributes, device=device)
            input_list.append(tensor)
        aa_embedding, lengths = utils.merge_tensors(input_list, device=device) # [bs, max_n_aa, 20]
        aa_embedding = self.aa_embeding_layer(aa_embedding) # [bs, max_n_aa, 20]->[bs, max_n_aa, h/2]
        aa_embedding = F.relu(aa_embedding)

        # return utils.de_merge_tensors(aa_embedding, lengths)
        return aa_embedding, lengths

    # @profile
    def forward(self, mapping: List[Dict], device):

        import time
        global starttime
        starttime = time.time()


        aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(n_aa, 20)]
        aa_indices = utils.get_from_mapping(mapping, 'aa_indices')
        atom_attributes = utils.get_from_mapping(mapping, 'atom_attributes') # List[List[np.ndarray=(n_atoms,)]]
        atom_indices = utils.get_from_mapping(mapping, 'atom_indices')

        batch_size = len(aa_attributes)

        # (bs, max(n_aa), h/2)
        aa_embedding, _ = self.attribute_embeding(aa_attributes, device, batch_size)

        # [bs, max(n_aa), h/2]
        element_states_batch, lane_states_batch =\
            self.forward_encode_sub_graph(
                mapping, 
                atom_attributes, atom_indices,
                device, batch_size)
        

        # [bs, max(n_aa), h], inputs_lengths: list of actual len_aa
        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)

        # create mask
        max_seq_num = max(inputs_lengths)
        # (bs, max_n_aa, max_n_aa)
        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)
        
        # TODO: try different method to aggregate features rather than concat
        # => [bs, max(n_aa), h]
        inputs = torch.cat([aa_embedding, inputs], dim=-1) 

        # global_graph: GlobalGraphRes [bs, max(n_aa), h]
        hidden_states = self.global_graph(inputs, attention_mask, mapping)
        
        # [bs, max(n_aa), h] => [bs, h]
        hidden_states = torch.max(hidden_states, dim=1)[0]

        # TODO: fine tune the structure of predicting score
        # [bs, h] => [bs, 1]
        outputs = self.reg_pred(hidden_states)

        #################################
        """
        matrix = utils.get_from_mapping(mapping, 'matrix')
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')

        batch_size = len(matrix)
        # for i in range(batch_size):
        # polyline_spans[i] = [slice(polyline_span[0], polyline_span[1]) for polyline_span in polyline_spans[i]]

        if args.argoverse:
            utils.batch_init(mapping)
        
        # 输出都是list[tensor]: subgraph全部特征[num_polylines,128], subgraph地图特征[n_lane_polylines,128]
        element_states_batch, lane_states_batch = self.forward_encode_sub_graph0(mapping, matrix, polyline_spans, device, batch_size)
        
        # 因为各polyline数量不同,需要padding: inputs:subgraph全部特征,shape=[batch,max_polylines,128], inputs_lengths:list[polyline实际数量]
        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1) # 因为用了padding，所以需要把不合法的polyline标记成0
        # hidden_states,shape=[batch,max_polylines,128]
        hidden_states = self.global_graph(inputs, attention_mask, mapping)

        utils.logging('time3', round(time.time() - starttime, 2), 'secs')

        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
        """
        return outputs