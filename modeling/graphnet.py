from typing import Dict, List, Tuple, NamedTuple, Any

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed
from modeling.loss import BinLoss, PredLoss
from modeling.decoder import Decoder
from axial_positional_embedding import AxialPositionalEmbedding
import utils

from modeling.layer import KMersNet, Conv2d

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics

# post_preprocessor

def is_pwm_type_valid(pwm_type):
    return pwm_type in ['hmm', 'pssm', 'psfm']

class SubGraph(nn.Module):
    # config:
    #   depth:int, hidden_size:int, point_level-4-3:bool
    def __init__(self, config, hidden_size, depth):
        super(SubGraph, self).__init__()
        
        # depth = config['depth']
        # hidden_size = config['hidden_size']
        self.hidden_size = hidden_size
        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) 
                                     for _ in range(depth)])
        
        # self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        if config['use_batch_norm']:
            self.layers_2 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(depth)])
        else:
            self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            
        #if config['point_level-4-3']:
        # self.layer_0_again = MLP(hidden_size)
        
        if config['use_atom_embedding']:
            atom_nfeatures = config['atom_nfeatures']
            self.atom_emb_layer = nn.Embedding(num_embeddings=atom_nfeatures + 1, 
                                               embedding_dim=atom_nfeatures)
            max_seq_len = max_num_atoms_in_aa + 5
            self.atom_pos_emb = AxialPositionalEmbedding(
                dim = atom_nfeatures,
                axial_shape = (math.ceil(max_seq_len / 64), 64))

            self.layer_0 = MLP(atom_nfeatures, hidden_size)
        
        self.config = config
        self.use_dropout = True
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

    # input_list:
    #   case1: a chain, shape=residues*[atoms, hidden] (att along atoms)
    #   case2: a batch of chains, shape=N*[residues, hidden] (att along residues)
    # def forward1(self, input_list : List[Tensor]):
    def forward(self, hidden_states : Tensor, lengths: List[int]):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        # input_list is list of tensor[nodes(atom/aa), hidden]
        # merge with padding: hidden_states = [lines, max_nodes, hidden]
        # hidden_states, lengths = utils.merge_tensors(input_list, device) # [aa, max_atoms, h]
        # hidden_states = input.unsqueeze(-1) # (aa, max(num_atoms), h)
        # hidden_size = hidden_states.shape[2] # here hidden size is input's
        max_vector_num = hidden_states.shape[1] # do att along this dim

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device) # atom case: [aa, max_atoms, max_atoms]
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            if hidden_states.type is not torch.long:
                hidden_states = hidden_states.type(torch.long) # (aa, max(num_atoms))
            hidden_states = self.atom_emb_layer(hidden_states) # (aa, max(num_atoms), e)
            hidden_states += self.atom_pos_emb(hidden_states)

        hidden_states = self.layer_0(hidden_states)
        # hidden_states = self.layer_0_again(hidden_states) # (aa, max(num_atoms), h/2)
        
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
            if self.use_dropout:
                hidden_states = self.dropout(hidden_states)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            
            if self.config['use_batch_norm']:
                orig_shape = hidden_states.shape
                hidden_states = self.layers_2[layer_index](hidden_states.reshape(-1, orig_shape[-1]))
                hidden_states = hidden_states.reshape(orig_shape)
            else:         
                hidden_states = self.layers_2[layer_index](hidden_states)
        
        if self.config['use_atom_embedding']:
            hidden_states = torch.max(hidden_states, dim=1)[0] # (aa, h)
        return hidden_states

    def forward2(self, input : Tensor, lengths: Tensor):
        device = input.device
        batch_size = input.shape[0]
        max_vector_num = input.shape[1]

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device)
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            if input.type is not torch.long:
                input = input.type(torch.long).squeeze(-1) # (N, seq,1) -> (N,seq,)
            hidden_states = self.atom_emb_layer(input)
            hidden_states += self.atom_pos_emb(hidden_states)

        hidden_states = self.layer_0(hidden_states)

        # if self.config['point_level-4-3']:
        hidden_states = self.layer_0_again(hidden_states)
        
        # mask sequence dim as each item in batch has difference seq length
        for i in range(batch_size):
            # assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        # list_hidden_states = []
        # for i in range(batch_size):
        #     iend = lengths[i]
        #     iend = max(1, iend)
        #     hidden_states_per_aa = torch.max(hidden_states[i, :iend], dim=0)[0]
        #     list_hidden_states.append(hidden_states_per_aa.unsqueeze(0))
        # hidden_states = torch.cat(list_hidden_states)
        
        return torch.max(hidden_states, dim=1)[0]
        # return hidden_states
        # return torch.max(hidden_states, dim=1)[0], \
        #        torch.cat(utils.de_merge_tensors(hidden_states, lengths))



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
        
        self.use_bin_label = args.label_bin
        self.use_sub_graph = True
        self.use_atom_graph = False

        sub_graph_hidden_dim = hidden_size // 2 if self.use_atom_graph else hidden_size
        if self.use_sub_graph:
            config['use_batch_norm'] = False
            config['point_level-4-3'] = True
            config['use_atom_embedding'] = True if self.use_atom_graph else False
            config['atom_nfeatures'] = len(list_atoms) # 38
            ### layers
            self.atom_level_sub_graph = SubGraph(config, sub_graph_hidden_dim, args.sub_graph_depth)
            self.laneGCN_A2L = CrossAttention(sub_graph_hidden_dim)
            self.laneGCN_L2L = GlobalGraphRes(sub_graph_hidden_dim)
            self.laneGCN_L2A = CrossAttention(sub_graph_hidden_dim)
            self.use_A2A = True
            if self.use_A2A:
                self.laneGCN_A2A = GlobalGraphRes(sub_graph_hidden_dim)

        self.global_graph = GlobalGraphRes(hidden_size)

        # TODO: adjust hidden size to 32 (used in ScanNet)
        # TODO: make 20 as param
        # TODO: make 2 as param
        # TODO: make 'relu' as para

        ##################################################################
        ### aa embedding
        ##################################################################
        self.pwm_type = args.pwm_type
        self.use_prot_chm = args.use_prot_chm_feature
        max_seq_len = 6000

        ### update output dim
        aa_emb_hidden_dim = sub_graph_hidden_dim if self.use_sub_graph else hidden_size
        self.aa_out_dim = aa_emb_hidden_dim
        if is_pwm_type_valid(self.pwm_type):
            self.aa_out_dim = aa_emb_hidden_dim // 2
            print("use prot pwm feature, output dim = {}".format(self.aa_out_dim))
        if self.use_prot_chm:
            self.aa_out_dim = aa_emb_hidden_dim // 2
            print("use prot chm feature, output dim = {}".format(self.aa_out_dim))
        ### seq embedding
        self.aa_input_features = 20
        self.aa_seq_layer = nn.Linear(in_features=self.aa_input_features, 
                                           out_features=self.aa_out_dim, 
                                           bias=False)
        self.aa_seq_layer = TimeDistributed(self.aa_seq_layer)
        # position embedding
        self.aa_pos_emb = AxialPositionalEmbedding(
            dim = self.aa_out_dim,
            axial_shape = (math.ceil(max_seq_len / 64), 64))
        self.na_seq_norm = LayerNorm(self.aa_out_dim)

        ### pwm embedding
        if is_pwm_type_valid(self.pwm_type):
            pwm_emd_dim = {'hmm': 30, 'pssm': 20, 'psfm': 20}
            self.aa_pwm_layer = nn.Linear(
                in_features=pwm_emd_dim[self.pwm_type], 
                out_features=self.aa_out_dim, 
                bias=False)
            self.aa_pwm_layer = TimeDistributed(self.aa_pwm_layer)
            # position embedding
            self.aa_pwm_pos_emb = AxialPositionalEmbedding(
                dim = self.aa_out_dim,
                axial_shape = (math.ceil(max_seq_len / 64), 64))
            self.na_pwm_norm = LayerNorm(self.aa_out_dim)
        
        ################ prot chemistry layer ################
        if self.use_prot_chm:
            self.prot_chm_dim = 37
            self.prot_chm_layer = nn.Linear(in_features=self.prot_chm_dim, 
                                            out_features=self.aa_out_dim, 
                                            bias=False)
            self.prot_chm_layer = TimeDistributed(self.prot_chm_layer)
            # position embedding
            self.aa_chm_pos_emb = AxialPositionalEmbedding(
                dim = self.aa_out_dim,
                axial_shape = (math.ceil(max_seq_len / 64), 64))
            self.na_chm_norm = LayerNorm(self.aa_out_dim)
        
        ##################################################################
        ### nc embedding
        ##################################################################
        self.use_nc_chm = args.use_chemistry
        self.use_kmers = False
        self.use_conv =  True
        na_max_seq_len = 1024

        nc_emb_hidden_dim = sub_graph_hidden_dim if self.use_sub_graph else hidden_size
        self.nc_out_dim = nc_emb_hidden_dim
        if self.use_nc_chm:
            self.nc_out_dim = nc_emb_hidden_dim // 2

        if self.use_kmers:
            self.kmers_net = KMersNet(base_channel=8, out_channels=self.nc_out_dim)
        elif self.use_conv: ## TODO: fix as current not allowed
            self.na_conv2d = Conv2d(1, 8, 
                                    kernel_size=(3, 5),
                                    bn=True, same_padding=True)
            self.na_fc = nn.Linear(in_features=32, out_features=self.nc_out_dim, bias=False) 
        else: ## TODO: fix as current not allowed
            self.nc_seq_layer = nn.Linear(in_features=4, out_features=self.nc_out_dim, bias=False)
            self.nc_seq_layer = TimeDistributed(self.nc_seq_layer)

        self.na_norm = LayerNorm(self.nc_out_dim)
        self.na_pos_emb = AxialPositionalEmbedding(
            dim = self.nc_out_dim,
            axial_shape = (math.ceil(na_max_seq_len / 64), 64))

        if self.use_nc_chm:
            # (num_nc, 1+9) => (num_nc, h/2)
            self.nc_chm_dim = 10
            self.nc_chm_layer = nn.Linear(in_features=self.nc_chm_dim, 
                                          out_features=self.nc_out_dim, 
                                          bias=False)
            # self.na_comb_layer = nn.Linear(in_features=hidden_size, out_features=sub_graph_hidden_dim, bias=False)
            # concat seq_feat and other_feat to (num_nc, h) => (num_nc, h/2)
            # nn.init.kaiming_uniform_(self.nc_chm_layer.weight, a=1)
            # nn.init.kaiming_uniform_(self.na_comb_layer.weight, a=1)

        # TODO: if we 
        self.reg_head = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)
        # self.reg_pred = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        if self.use_bin_label:
            # TODO: get from parameters
            num_bins = 8
            ### logics and regression header
            self.head_logics = nn.Linear(in_features=hidden_size, out_features=num_bins, bias=False)
            self.head_reg = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
            ### decoder and loss function
            self.decoder = Decoder(args)
            self.loss = BinLoss(config)
            
            nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
            # TODO: init weight and bias
            nn.init.normal_(self.head_reg.weight, std=0.001)
            nn.init.normal_(self.head_logics.weight, std=0.001)
            for l in [self.reg_head, self.head_logics, self.head_reg]:
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)
        else:
            self.reg_pred = nn.Linear(in_features=hidden_size, out_features=1, bias=False) 
            self.loss = PredLoss(config)
            
            # use detectron2's box-head-wgt-init method to init fc's weight
            nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
            # TODO: init weight and bias
            # nn.init.normal_(self.reg_head.weight, std=0.001)
            nn.init.normal_(self.reg_pred.weight, std=0.001)
            for l in [self.reg_head, self.reg_pred]:
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)
       
        self.use_dropout = True
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1) 
        

    def forward_encode_sub_graph2(
            self, 
            atom_attributes, # List[np.ndarray]
            aa_lengths, # List[List[Int]]
            nc_embedding,
            nc_lengths, # [List[Int]]
            device, batch_size) -> Tuple[Tensor, Tensor]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        list_atom_attributes = []
        for it in atom_attributes:
            it_tensor = torch.tensor(it, device=device)
            list_atom_attributes.append(it_tensor)
        rearraged_data, prot_lengths \
            = utils.merge_tensors(list_atom_attributes, device) # (N, max(aa), max(atoms))
        rearraged_data = rearraged_data.reshape(-1, rearraged_data.shape[2], 1)
        padding_aa_lengths = torch.zeros([batch_size, max(prot_lengths)], dtype=torch.long, device=device) 
        for i in range(batch_size):
            for j in range(prot_lengths[i]):
                padding_aa_lengths[i,j] = aa_lengths[i][j]
        padding_aa_lengths = padding_aa_lengths.reshape(-1) # (N * max(aa),)

        aa_embedding = self.atom_level_sub_graph(rearraged_data, padding_aa_lengths) # (N*max(num_aa), 64)
        aa_embedding = aa_embedding.reshape(batch_size, -1, aa_embedding.shape[-1]) # (N, max(num_aa), 64)

        # TODO: decide if we should use subgraph of rdna atoms
        # query => key : DNA/RNA => protein
        attention_mask = torch.zeros([batch_size, nc_embedding.shape[1], aa_embedding.shape[1]], device=device)
        for i in range(batch_size):
            attention_mask[i, :nc_lengths[i], :prot_lengths[i]].fill_(1)
        nc_embedding = nc_embedding + self.laneGCN_A2L(nc_embedding, aa_embedding, attention_mask)
        
        # DNA/RNA => DNA/RNA
        attention_mask = torch.zeros([batch_size, nc_embedding.shape[1], nc_embedding.shape[1]], device=device)
        for i in range(batch_size):
            attention_mask[i, :nc_lengths[i], :nc_lengths[i]].fill_(1)
        nc_embedding = nc_embedding + self.laneGCN_L2L(nc_embedding, attention_mask)
        
        # protein => DNA/RNA
        attention_mask = torch.zeros([batch_size, aa_embedding.shape[1], nc_embedding.shape[1]], device=device)
        for i in range(batch_size):
            attention_mask[i, :prot_lengths[i], :nc_lengths[i]].fill_(1)
        aa_embedding = aa_embedding + self.laneGCN_L2A(aa_embedding, nc_embedding, attention_mask)
        
        # protein => protein
        if self.use_A2A:
            attention_mask = torch.zeros([batch_size, aa_embedding.shape[1], aa_embedding.shape[1]], device=device)
            for i in range(batch_size):
                attention_mask[i, :prot_lengths[i], :prot_lengths[i]].fill_(1)
            aa_embedding = aa_embedding + self.laneGCN_A2A(aa_embedding, attention_mask)

        # => (N, max(num_aa)+max(num_nc), h/2)
        # list_complex_embedding = []
        # for i in range(batch_size):
        #     # => (max(num_aa)+max(num_nc), h/2)
        #     temp = torch.cat([aa_embedding[i,:prot_lengths[i]], nc_embedding[i,:nc_lengths[i]]], dim=0)
        #     list_complex_embedding.append(temp)

        # return list_complex_embedding # List[Tensor=(max(num_aa)+max(num_nc), h/2)]
        return aa_embedding, nc_embedding

    def forward_encode_sub_graph(
            self, 
            atom_attributes, # List[np.ndarray]
            aa_lengths, # List[List[Int]]
            nc_embedding,
            nc_lengths, # [List[Int]]
            device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        """
        prot_states_batch = []
        input_list = []
        for i in range(batch_size):
            tensor = torch.tensor(atom_attributes[i], device=device) # (num_aa, MAX_NUM_ATOMS)
            input_list.append(tensor)
        
        for i in range(batch_size):
            a = self.atom_level_sub_graph(input_list[i], aa_lengths[i]) # (num_aa, h/2)
            prot_states_batch.append(a)
        
        for i in range(batch_size):
            prot_feats = prot_states_batch[i] # (num_aa, h/2)
            rdna_feats = nc_embedding[i, :nc_lengths[i]] # (num_nc, h/2)

            rdna_feats = rdna_feats + self.laneGCN_A2L(
                rdna_feats.unsqueeze(0), prot_feats.unsqueeze(0)).squeeze(0)
            # if self.use_dropout:
            #     rdna_feats = self.dropout(rdna_feats)
            
            rdna_feats = rdna_feats + self.laneGCN_L2L(
                rdna_feats.unsqueeze(0)).squeeze(0)
            if self.use_dropout:
                rdna_feats = self.dropout(rdna_feats)
                
            prot_feats = prot_feats + self.laneGCN_L2A(
                prot_feats.unsqueeze(0), rdna_feats.unsqueeze(0)).squeeze(0)
            # if self.use_dropout:
            #     prot_feats = self.dropout(prot_feats)
                
            if self.use_A2A:
                prot_feats = prot_feats + self.laneGCN_A2A(
                    prot_feats.unsqueeze(0)).squeeze(0)
            if self.use_dropout:
                prot_feats = self.dropout(prot_feats)
                
            prot_states_batch[i] = torch.cat([prot_feats, rdna_feats]) # [num_aa+num_nc, h/2]
            # prot_states_batch[i] = prot_feats # [aa, h/2]
            
            # if self.use_dropout:
            #     prot_states_batch[i] = self.dropout(prot_states_batch[i])

        return prot_states_batch

    def forward_encode_sub_graph_updated(
            self, 
            prot_embedding,  # Tensor
            prot_lengths,    # List[Int]
            nc_embedding,    # Tensor
            nc_lengths,      # [List[Int]]
            device, batch_size) -> List[Tensor]:
        x1 = self.atom_level_sub_graph(prot_embedding, prot_lengths) # (bs, max num_aa, h/2)
        x2 = self.atom_level_sub_graph(nc_embedding, nc_lengths) # (bs, max num_nc, h/2)
        
        list_emb = []
        for i in range(batch_size):
            # prot_feats = prot_states_batch[i] # (num_aa, h/2)
            prot_feats = x1[i, :prot_lengths[i]] # (num_aa, h/2)
            rdna_feats = x2[i, :nc_lengths[i]] # (num_nc, h/2)

            rdna_feats = rdna_feats + self.laneGCN_A2L(
                rdna_feats.unsqueeze(0), prot_feats.unsqueeze(0)).squeeze(0)
            rdna_feats = rdna_feats + self.laneGCN_L2L(
                rdna_feats.unsqueeze(0)).squeeze(0)
            if self.use_dropout:
                rdna_feats = self.dropout(rdna_feats)
                
            prot_feats = prot_feats + self.laneGCN_L2A(
                prot_feats.unsqueeze(0), rdna_feats.unsqueeze(0)).squeeze(0)
            if self.use_A2A:
                prot_feats = prot_feats + self.laneGCN_A2A(
                    prot_feats.unsqueeze(0)).squeeze(0)
            if self.use_dropout:
                prot_feats = self.dropout(prot_feats)

            tmp = torch.cat([prot_feats, rdna_feats]) # [num_aa+num_nc, h/2]
            list_emb.append(tmp)

        return list_emb
    
    def aa_attribute_embeding(
            self,
            aa_attributes : List[np.ndarray], 
            aa_pwm : List[np.ndarray],
            aa_chm : List[np.ndarray],
            device, batch_size) -> Tuple[Tensor, List[int]]:
        """
        compute embedding of amino acid
        inputs:
            aa_attributes: List[Tensor=(n_aa, 20)]
        outputs:
            Tensor=(N, max(num_aa), h/2)
            List[int], each prot length
        """
        prot_list = []
        for i in range(batch_size):
            prot = torch.tensor(aa_attributes[i], device=device)
            prot_list.append(prot)
        x, lengths = utils.merge_tensors(prot_list, device=device) # [bs, max_n_aa, 20]
        x = self.aa_seq_layer(x) # [bs, max_n_aa, 20]->[bs, max_n_aa, h/4]
        x += self.aa_pos_emb(x)
        # x = self.dropout(x)
        x = self.na_seq_norm(x)
        
        if is_pwm_type_valid(self.pwm_type) and aa_pwm is not None:
            pwm_list = []
            for i in range(batch_size):
                tensor = torch.tensor(aa_pwm[i], device=device)
                pwm_list.append(tensor)
            aa_pwm_embedding, _ = utils.merge_tensors(pwm_list, device=device) # [bs, max_n_aa, 30]
            aa_pwm_embedding = self.aa_pwm_layer(aa_pwm_embedding) # [bs, max_n_aa, 30]->[bs, max_n_aa, h/4]
            aa_pwm_embedding += self.aa_pwm_pos_emb(aa_pwm_embedding)
            # aa_pwm_embedding = self.dropout(aa_pwm_embedding)
            aa_pwm_embedding = self.na_pwm_norm(aa_pwm_embedding)
            # aa_pwm_embedding = F.relu(aa_pwm_embedding)
            x = torch.cat([x, aa_pwm_embedding], dim=-1) # [bs, max_n_aa, h/2]

        if self.use_prot_chm:
            chm_list = []
            for i in range(batch_size):
                tensor = torch.tensor(aa_chm[i], device=device)
                chm_list.append(tensor)
            chm_embedding, _ = utils.merge_tensors(chm_list, device=device) # (bs, max(num_aa), 37)
            chm_embedding = self.prot_chm_layer(chm_embedding) # (bs, max(num_aa), h/4)
            chm_embedding += self.aa_chm_pos_emb(chm_embedding)
            # chm_embedding = self.dropout(chm_embedding)
            chm_embedding = self.na_chm_norm(chm_embedding)
            # chm_embedding = F.relu(chm_embedding)
            x = torch.cat([x, chm_embedding], dim=-1) # (bs, max(num_aa), h/2)
            
        x = F.relu(x)
        return x, lengths

    def na_attribute_embeding(
            self,
            na_attributes : List[np.ndarray], 
            na_other_attributes : List[np.ndarray], 
            device, batch_size) -> Tuple[Tensor, List[int]]:
        """
        compute embedding of Nucleic Acids
        inputs:
            na_attributes: List[Tensor=(n_na, 4)]
        outputs:
            List[Tensor=(n_na, h/2)] 
        """
        input_list = []
        for i in range(batch_size):
            tensor = torch.tensor(na_attributes[i], device=device)
            input_list.append(tensor)
        x, lengths = utils.merge_tensors(input_list, device=device) # (bs, max(n_na), 4)

        if self.use_kmers:
            x = self.kmers_net(x.unsqueeze(1)) # (bs, 1, max(n_na), 4)->(bs, max(n_na), h)
        elif self.use_conv:
            x = self.na_conv2d(x.unsqueeze(1)) # (N,1,seq,4)=>(N,C,seq,4)
            x = x.permute(0,2,1,3).contiguous() # (N,Seq,C,4)
            x = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) # (N,Seq,C*4)
            x = self.na_fc(x) # (N,Seq,h)
            
        else:
            x = self.nc_seq_layer(x) # (bs, max(n_na), 4)->(bs, max(n_na), h)
        x += self.na_pos_emb(x)

        if self.use_nc_chm and na_other_attributes is not None:
            chm_list = []
            for i in range(batch_size):
                tensor = torch.tensor(na_other_attributes[i], device=device)
                chm_list.append(tensor)
            chm_emb, _ = utils.merge_tensors(chm_list, device=device) # (bs, max(n_na), 10)
            chm_emb = self.nc_chm_layer(chm_emb) # (bs, max(n_na), h/2)
            x = torch.cat([x, chm_emb], dim=-1) # (bs, max(n_na), h)
        
        x = self.na_norm(x) # (N,Seq,h)
        x = F.relu(x)
        return x, lengths

    def forward(self, mapping: List[Dict], device):
        ### amino acid features
        aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(num_aa, 20)]
        aa_pwm = utils.get_from_mapping(mapping, 'aa_pwm') if is_pwm_type_valid(self.pwm_type) else None
        aa_chm = utils.get_from_mapping(mapping, 'aa_chm') if self.use_prot_chm else None
        atom_attributes = utils.get_from_mapping(mapping, 'atom_attributes') # List[np.ndarray=(num_aa, MAX_NUM_ATOMS)]
        aa_lengths = utils.get_from_mapping(mapping, 'aa_lengths') # List[List[int]]

        ### nuc acid features
        na_attributes = utils.get_from_mapping(mapping, 'nucleotide_attributes') # List[np.ndarray=(num_nc, 4)]
        # List[np.ndarray=(n_nc, 10)]
        na_other_attributes = utils.get_from_mapping(mapping, 'nucleotide_other_attributes') if self.use_nc_chm else None

        batch_size = len(aa_attributes)

        # compute embedding feature for amino acid sequence, (N, max(num_aa), h/2)
        aa_embedding, prot_lengths = self.aa_attribute_embeding(aa_attributes, aa_pwm, aa_chm, device, batch_size)
        # compute embedding feature for DNA/RNA, (N, max(num_nc), h/2)
        na_embedding, nc_lengths = self.na_attribute_embeding(na_attributes, na_other_attributes, device, batch_size)
        
        ########################################################
        if self.use_sub_graph and not self.use_atom_graph:
            complex_embedding = self.forward_encode_sub_graph_updated(
                aa_embedding, prot_lengths, na_embedding, nc_lengths, device, batch_size)
            complex_embedding, complex_lengths = utils.merge_tensors(complex_embedding, device=device)

        elif self.use_sub_graph and self.use_atom_graph:
            complex_embedding = self.forward_encode_sub_graph(
                atom_attributes, aa_lengths, na_embedding, nc_lengths, device, batch_size)
            complex_embedding, complex_lengths = utils.merge_tensors(complex_embedding, device=device)
            ### complex_embedding
            aa_embedding = utils.de_merge_tensors(aa_embedding, prot_lengths)
            na_embedding = utils.de_merge_tensors(na_embedding, nc_lengths)
            complex_embedding_from_seq = []
            for f1, f2 in zip(aa_embedding, na_embedding):
                f3 = torch.cat([f1, f2], dim=0) # => (num_aa+num_nc, h/2)
                complex_embedding_from_seq.append(f3)
            # => (bs, max(num_aa+num_nc), h/2)
            complex_embedding_from_seq, _ = utils.merge_tensors(complex_embedding_from_seq, device=device)
            complex_embedding = torch.cat([complex_embedding, complex_embedding_from_seq], dim=-1)   

            ### do padding for sub graph (but very slow)
            '''
            # [bs, max(n_aa+n_nc), h/2]
            aa_embedding_from_atom, nc_embedding_from_atom = self.forward_encode_sub_graph(
                atom_attributes, aa_lengths, na_embedding, nc_lengths, device, batch_size)

            complex_lengths = [l1+l2 for l1, l2 in zip(prot_lengths, nc_lengths)]

            # [bs, max(n_aa+n_nc), h/2], complex_lengths: list of actual len_aa+len_nc of each batch
            # embedding_from_atom, complex_lengths = utils.merge_tensors(embedding_from_atom, device=device)

            assert(aa_embedding_from_atom.shape[1] == aa_embedding.shape[1])
            assert(nc_embedding_from_atom.shape[1] == na_embedding.shape[1])
            aa_embedding = torch.cat([aa_embedding_from_atom, aa_embedding], dim=-1) # => (N, max(num_aa), h)
            na_embedding = torch.cat([nc_embedding_from_atom, na_embedding], dim=-1)

            complex_embedding = []
            for i in range(batch_size):
                temp = torch.cat([aa_embedding[i, :prot_lengths[i]], 
                                na_embedding[i, :nc_lengths[i]]], dim=0) # (n_aa+n_nc, h)
                complex_embedding.append(temp) # (n_aa+n_nc, h)
            # complex_embedding = torch.cat(complex_embedding, dim=0) # (N, n_aa+n_nc, h)
            complex_embedding, _ = utils.merge_tensors(complex_embedding, device=device) # (N, n_aa+n_nc, h)
            '''
        else:
            aa_embedding = utils.de_merge_tensors(aa_embedding, prot_lengths)
            na_embedding = utils.de_merge_tensors(na_embedding, nc_lengths)
            complex_embedding_from_seq = []
            for f1, f2 in zip(aa_embedding, na_embedding):
                f3 = torch.cat([f1, f2], dim=0) # => (num_aa+num_nc, h/2)
                complex_embedding_from_seq.append(f3)
            complex_embedding, complex_lengths \
                = utils.merge_tensors(complex_embedding_from_seq, device=device)

        # [bs, max(n_aa+n_nc), h/2], inputs_lengths: list of actual len_aa+len_nc of each batch
        # embedding_from_seq, _ = utils.merge_tensors(embedding_from_seq, device=device)
        # [bs, max(n_aa+n_nc), h/2], [bs, max(n_aa+n_nc), h/2]=> [bs, max(n_aa+n_nc), h]
        # inputs = torch.cat([embedding_from_seq, embedding_from_atom], dim=-1)
        assert(len(prot_lengths) == len(nc_lengths))
        complex_lengths = [lp + ln for lp, ln in zip(prot_lengths, nc_lengths)]
        max_seq_num = max(complex_lengths)
        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device)
        for i, length in enumerate(complex_lengths):
            attention_mask[i, :length, :length].fill_(1)

        # global_graph: GlobalGraphRes [bs, max(n_aa), h]
        hidden_states = self.global_graph(complex_embedding, attention_mask, mapping)
        
        # to list[complex_feat], list number = batch size
        hidden_state_per_complex = utils.de_merge_tensors(hidden_states, complex_lengths)
        combined_feat_per_complex = []
        for i in range(batch_size):
            len_aa = prot_lengths[i]
            seq_feat_aa = hidden_state_per_complex[i][:len_aa] # (n_aa, h)
            seq_feat_nc = hidden_state_per_complex[i][len_aa:] # (n_nc, h)
            unit_feat_aa = torch.max(seq_feat_aa, dim=0)[0] # (h,)
            unit_feat_nc = torch.max(seq_feat_nc, dim=0)[0] # (h,)
            combined_feat_per_complex.append( torch.cat([unit_feat_aa, unit_feat_nc]).unsqueeze(0) ) # (1, h*2)
        hidden_states = torch.cat(combined_feat_per_complex, dim=0) # => (bs, h*2)

        # hidden_states = torch.max(hidden_states, dim=1)[0] # [bs, max(n_aa), h] => [bs, h]
        # TODO: fine tune the structure of predicting score
        
        hidden_states = self.reg_head(hidden_states) # (bs, h*2)=>(bs, h)

        labels = utils.get_from_mapping(mapping, 'label')
        labels = torch.tensor(labels, device=device, dtype=torch.float32).reshape(-1, 1) # [bs, 1]

        if self.use_bin_label:
            # get label and offset
            
            # bin_ctrs = bin_ctrs,
            # bin_half_w = bin_half_w,
            # label_bin = label_bin, # int
            # label_offset = label_offset,
            bin_ctrs    = np.array( utils.get_from_mapping(mapping, 'bin_ctrs') )       # (bs, n_bins)
            bin_half_w  = np.array( utils.get_from_mapping(mapping, 'bin_half_w') )     # (bs, n_bins)
            gt_bins     = np.array( utils.get_from_mapping(mapping, 'label_bin') ).reshape(-1,1)      # (bs, 1) int
            gt_offsets  = np.array( utils.get_from_mapping(mapping, 'label_offset') ).reshape(-1,1)    # (bs, 1)


            gt_bins     = torch.tensor(gt_bins, device=device, dtype=torch.int64)
            gt_offsets  = torch.tensor(gt_offsets, device=device, dtype=torch.float32)
            bin_ctrs  = torch.tensor(bin_ctrs, device=device, dtype=torch.float32)
            bin_half_w  = torch.tensor(bin_half_w, device=device, dtype=torch.float32)

            ### header
            logics_hidden = self.head_logics(hidden_states) # (bs, h) -> (bs, num_bins)
            reg_hidden = self.head_reg(hidden_states) # (bs, h) -> (bs, 1)
            
            ### decode
            outputs = self.decoder(gt_bins, gt_offsets, bin_ctrs, bin_half_w,
                                   logics_hidden, reg_hidden) # (bs, 1)
            
            gt_logics = gt_bins # torch.tensor(gt_bins, device=device, dtype=torch.int64).reshape(-1)        # (bs,)
            gt_reg    = gt_offsets #.reshape(-1,1) # torch.tensor(gt_reg, device=device, dtype=torch.float32).reshape(-1, 1)   # (bs, 1)
            
            ###
            loss_output = self.loss(
                logics_hidden, gt_logics.reshape(-1), 
                reg_hidden, gt_reg)
            
            

        else:
            outputs = self.reg_pred(hidden_states) # [bs, h] => [bs, 1]
            loss_output = self.loss(outputs, labels)

        return loss_output['loss'], outputs, loss_output


