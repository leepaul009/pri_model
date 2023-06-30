from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes, TimeDistributed
import utils

from modeling.layer import KMersNet

from preprocessing.protein_chemistry import list_atoms

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics

# post_preprocessor
import os

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
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        
        #if config['point_level-4-3']:
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

        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states) # (aa, max(num_atoms), h/2)
        
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

        return torch.max(hidden_states, dim=1)[0] # (aa, h) # , torch.cat(utils.de_merge_tensors(hidden_states, lengths))

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

class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        # self.loss = nn.MSELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor):
        """
        pred=(N, 1)
        target=(N, 1)
        """
        output = dict()
        num_reg = pred.shape[0]
        loss = self.reg_loss(pred, target)
        output['num_reg'] = num_reg
        output['reg_loss'] = loss
        output['loss'] = loss / (num_reg + 1e-10)
        return output

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
        atom_nfeatures = len(list_atoms) # 38
        self.atom_emb_layer = nn.Embedding(num_embeddings=atom_nfeatures + 1, 
                                           embedding_dim=args.hidden_size)
        
        # config['depth'] = args.sub_graph_depth
        # config['hidden_size'] = args.hidden_size
        config['point_level-4-3'] = True
        config['use_atom_embedding'] = True
        config['atom_nfeatures'] = atom_nfeatures
        self.atom_level_sub_graph = SubGraph(config, hidden_size // 2, args.sub_graph_depth)
        # self.point_level_cross_attention = CrossAttention(hidden_size)

        # self.global_graph = GlobalGraph(hidden_size)
        # output dim = hidden_size
        self.global_graph = GlobalGraphRes(hidden_size)
        
        self.laneGCN_A2L = CrossAttention(hidden_size // 2)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size // 2)
        self.laneGCN_L2A = CrossAttention(hidden_size // 2)
        self.use_A2A = True
        if self.use_A2A:
            self.laneGCN_A2A = GlobalGraphRes(hidden_size // 2)

        # TODO: adjust hidden size to 32 (used in ScanNet)
        # TODO: make 20 as param
        # TODO: make 2 as param
        # TODO: make 'relu' as para
        # self.aa_embeding_layer = nn.RNN(
        #     input_size=20, hidden_size=hidden_size // 2, num_layers=2, 
        #     nonlinearity='relu', batch_first=True)
        self.pwm_type = args.pwm_type

        self.aa_output_features = hidden_size // 2
        if is_pwm_type_valid(self.pwm_type):
            self.aa_output_features = hidden_size // 4

        self.aa_input_features = 20
        self.aa_embeding_layer = nn.Linear(in_features=self.aa_input_features, out_features=self.aa_output_features, bias=False)
        self.aa_embeding_layer = TimeDistributed(self.aa_embeding_layer)
        
  
        if self.pwm_type == 'hmm':
            self.aa_pwm_features = 30
            self.aa_pwm_embeding_layer = nn.Linear(in_features=self.aa_pwm_features, out_features=hidden_size // 4, bias=False)
            self.aa_pwm_embeding_layer = TimeDistributed(self.aa_pwm_embeding_layer)
        elif self.pwm_type == 'pssm':
            self.aa_pwm_features = 20
            self.aa_pwm_embeding_layer = nn.Linear(in_features=self.aa_pwm_features, out_features=hidden_size // 4, bias=False)
            self.aa_pwm_embeding_layer = TimeDistributed(self.aa_pwm_embeding_layer)
        elif self.pwm_type == 'psfm':
            self.aa_pwm_features = 20
            self.aa_pwm_embeding_layer = nn.Linear(in_features=self.aa_pwm_features, out_features=hidden_size // 4, bias=False)
            self.aa_pwm_embeding_layer = TimeDistributed(self.aa_pwm_embeding_layer)
        else:
            NotImplemented

        self.use_chemistry = args.use_chemistry

        self.use_conv = False
        self.use_kmers = True
        if self.use_kmers:
            out_channels = hidden_size // 2
            if self.use_chemistry:
                out_channels = hidden_size // 4
            self.kmers_net = KMersNet(base_channel=8, out_channels=out_channels)
        elif self.use_conv: ## TODO: fix as current not allowed
            # w=seq,h=4 ==> output: w'=seq-2(seq-3+1), h'=1(4-4+1)
            # h-kh+1:, w-kw+1:
            self.na_conv2d = nn.Conv2d(in_channels=1, out_channels=hidden_size // 2, kernel_size=(3,4))
        else: ## TODO: fix as current not allowed
            self.na_embeding_layer = nn.Linear(in_features=4, out_features=hidden_size // 2, bias=False)
            self.na_embeding_layer = TimeDistributed(self.na_embeding_layer)

        if self.use_chemistry:
            # (num_nc, 1+9) => (num_nc, h/2)
            self.na_chemi_layer = nn.Linear(in_features=10, out_features=hidden_size // 4, bias=False)
            # self.na_comb_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size // 2, bias=False)
            # concat seq_feat and other_feat to (num_nc, h) => (num_nc, h/2)
            # nn.init.kaiming_uniform_(self.na_chemi_layer.weight, a=1)
            # nn.init.kaiming_uniform_(self.na_comb_layer.weight, a=1)

        # TODO: if we 
        self.reg_head = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=False)
        self.reg_pred = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        # use detectron2's box-head-wgt-init method to init fc's weight

        nn.init.kaiming_uniform_(self.reg_head.weight, a=1)
        # TODO: init weight and bias
        # nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.normal_(self.reg_pred.weight, std=0.001)
        for l in [self.reg_head, self.reg_pred]:
            if l.bias is not None:
                nn.init.constant_(l.bias, 0)
        
        self.loss = PredLoss(config)

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
            rdna_feats = rdna_feats + self.laneGCN_L2L(
                rdna_feats.unsqueeze(0)).squeeze(0)
            prot_feats = prot_feats + self.laneGCN_L2A(
                prot_feats.unsqueeze(0), rdna_feats.unsqueeze(0)).squeeze(0)
            
            if self.use_A2A:
                prot_feats = prot_feats + self.laneGCN_A2A(
                    prot_feats.unsqueeze(0)).squeeze(0)

            prot_states_batch[i] = torch.cat([prot_feats, rdna_feats]) # [num_aa+num_nc, h/2]
            # prot_states_batch[i] = prot_feats # [aa, h/2]

        return prot_states_batch
    
    def aa_attribute_embeding(
            self,
            aa_attributes : List[np.ndarray], 
            aa_pwm : List[np.ndarray],
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
        aa_embedding, lengths = utils.merge_tensors(prot_list, device=device) # [bs, max_n_aa, 20]
        aa_embedding = self.aa_embeding_layer(aa_embedding) # [bs, max_n_aa, 20]->[bs, max_n_aa, h/4]
        aa_embedding = F.relu(aa_embedding)

        if is_pwm_type_valid(self.pwm_type) and aa_pwm is not None:
            pwm_list = []
            for i in range(batch_size):
                tensor = torch.tensor(aa_pwm[i], device=device)
                pwm_list.append(tensor)
            aa_pwm_embedding, _ = utils.merge_tensors(pwm_list, device=device) # [bs, max_n_aa, 30]
            aa_pwm_embedding = self.aa_pwm_embeding_layer(aa_pwm_embedding) # [bs, max_n_aa, 30]->[bs, max_n_aa, h/4]
            aa_pwm_embedding = F.relu(aa_pwm_embedding)    
            final_embedding = torch.cat([aa_embedding, aa_pwm_embedding], dim=-1) # [bs, max_n_aa, h/2]
        else:
            final_embedding = aa_embedding

        # return utils.de_merge_tensors(final_embedding, lengths), lengths
        return final_embedding, lengths

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
        # merge to Tensor=(bs, max(n_na), 4)
        na_embedding, lengths = utils.merge_tensors(input_list, device=device)

        if self.use_kmers:
            # (bs, 1, max(n_na), 4)->(bs, max(n_na), h/2)
            na_embedding = self.kmers_net(na_embedding.unsqueeze(1))
        elif self.use_conv:
            # (bs, 1, max(n_na), 4)->(bs, h/2, max(n_na)-2, 1)
            temp = self.na_conv2d(na_embedding.unsqueeze(1))
            # (bs, h/2, max(n_na)-2, 1) -> (bs, max(n_na)-2, h/2)
            na_embedding = temp.squeeze(-1).permute(0, 2, 1).contiguous()
            lengths = [max(l-2, 1) for l in lengths]
        else:
            # (bs, max(n_na), 4)->(bs, max(n_na), h/2)
            na_embedding = self.na_embeding_layer(na_embedding)

        if self.use_chemistry and na_other_attributes is not None:
            chemi_input_list = []
            for i in range(batch_size):
                chemi_tensor = torch.tensor(na_other_attributes[i], device=device)
                chemi_input_list.append(chemi_tensor)
            # merge to Tensor=(bs, max(n_na), 10)
            na_chemi_embedding, _ = utils.merge_tensors(chemi_input_list, device=device)
            # (bs, max(n_na), 10)->(bs, max(n_na), h/4)
            na_chemi_embedding = self.na_chemi_layer(na_chemi_embedding)
            na_embedding = torch.cat([na_embedding, na_chemi_embedding], dim=-1) # (bs, max(n_na), h/2)
            # na_embedding = self.na_comb_layer(na_embedding) # (bs, max(n_na), h) => (bs, max(n_na), h/2)

        na_embedding = F.relu(na_embedding)

        # return utils.de_merge_tensors(na_embedding, lengths), lengths
        return na_embedding, lengths

    def forward(self, mapping: List[Dict], device):
        # amino acid feature, tensor shape = (number_of_amino_acid, 20)
        aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(n_aa, 20)]
        aa_pwm = None
        if is_pwm_type_valid(self.pwm_type):
            aa_pwm = utils.get_from_mapping(mapping, 'aa_pwm')

        atom_attributes = utils.get_from_mapping(mapping, 'atom_attributes') # List[np.ndarray=(num_aa, MAX_NUM_ATOMS)]
        aa_lengths = utils.get_from_mapping(mapping, 'aa_lengths') # List[List[int]]
        # aa_indices = utils.get_from_mapping(mapping, 'aa_indices') # not used
        # atom_indices = utils.get_from_mapping(mapping, 'atom_indices')
        # DNA/RNA feature, tensor shape = (number_of_DNA/RNA, 4)
        na_attributes = utils.get_from_mapping(mapping, 'nucleotide_attributes') # List[np.ndarray=(num_nc, 4)]

        if self.use_chemistry:
            na_other_attributes = utils.get_from_mapping(mapping, 'nucleotide_other_attributes') # List[np.ndarray=(n_nc, 10)]
        else:
            na_other_attributes = None

        batch_size = len(aa_attributes)

        # compute embedding feature for amino acid sequence, (N, max(num_aa), h/2)
        aa_embedding, prot_lengths = self.aa_attribute_embeding(aa_attributes, aa_pwm, device, batch_size)
        # compute embedding feature for DNA/RNA, (N, max(num_nc), h/2)
        na_embedding, nc_lengths = self.na_attribute_embeding(na_attributes, na_other_attributes, device, batch_size)
        
        ########################################################
        
        complex_embedding = self.forward_encode_sub_graph(
            atom_attributes, aa_lengths, na_embedding, nc_lengths, device, batch_size)

        complex_embedding, complex_lengths = utils.merge_tensors(complex_embedding, device=device)
        
        max_seq_num = max(complex_lengths)
        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device)
        for i, length in enumerate(complex_lengths):
            attention_mask[i, :length, :length].fill_(1)

        # complex_embedding
        aa_embedding = utils.de_merge_tensors(aa_embedding, prot_lengths)
        na_embedding = utils.de_merge_tensors(na_embedding, nc_lengths)
        complex_embedding_from_seq = []
        for f1, f2 in zip(aa_embedding, na_embedding):
            f3 = torch.cat([f1, f2], dim=0)
            complex_embedding_from_seq.append(f3)
        complex_embedding_from_seq, _ = utils.merge_tensors(complex_embedding_from_seq, device=device)
        complex_embedding = torch.cat([complex_embedding, complex_embedding_from_seq], dim=-1)
        
        ########################################################
        
        ### do padding for sub graph (but very slow)
        '''
        # [bs, max(n_aa+n_nc), h/2]
        aa_embedding_from_atom, nc_embedding_from_atom = self.forward_encode_sub_graph(
            atom_attributes, aa_lengths, na_embedding, nc_lengths, device, batch_size)

        complex_lengths = [l1+l2 for l1, l2 in zip(prot_lengths, nc_lengths)]

        # [bs, max(n_aa+n_nc), h/2], complex_lengths: list of actual len_aa+len_nc of each batch
        # embedding_from_atom, complex_lengths = utils.merge_tensors(embedding_from_atom, device=device)

        # create mask
        max_seq_num = max(complex_lengths)
        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device) # (bs, max_n_aa, max_n_aa)
        for i, length in enumerate(complex_lengths):
            attention_mask[i, :length, :length].fill_(1)
        
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
        
        ########################################################


        # [bs, max(n_aa+n_nc), h/2], inputs_lengths: list of actual len_aa+len_nc of each batch
        # embedding_from_seq, _ = utils.merge_tensors(embedding_from_seq, device=device)
        # [bs, max(n_aa+n_nc), h/2], [bs, max(n_aa+n_nc), h/2]=> [bs, max(n_aa+n_nc), h]
        # inputs = torch.cat([embedding_from_seq, embedding_from_atom], dim=-1) 

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
        outputs = self.reg_pred(hidden_states) # [bs, h] => [bs, 1]
         
        labels = utils.get_from_mapping(mapping, 'label')
        labels = torch.tensor(labels, device=device, dtype=torch.float32).reshape(-1, 1) # [bs, 1]

        loss_output = self.loss(outputs, labels)

        return loss_output['loss'], outputs, loss_output


class PostProcess(nn.Module):
    def __init__(self, output_dir):
        super(PostProcess, self).__init__()
        self.output_dir = output_dir

    def forward(self, out):
        post_out = dict()
        post_out['num_reg'] = out['num_reg']
        post_out['reg_loss'] = out['reg_loss']
        return post_out

    def append(self, metrics: Dict, post_out=None, preds=None, input=None) -> Dict:
        
        if len(metrics.keys()) == 0:
            for key in post_out:
                if key != "loss":
                    metrics[key] = 0.0
        
        # num_reg, re_loss
        for key in post_out:
            if key == 'reg_loss':
                metrics[key] += post_out[key].item()
            else:
                metrics[key] += post_out[key]
            # print("post process: {} = {}".format(key, metrics[key]))

        if preds is not None and input is not None:
            if "preds" not in metrics:
                preds = preds.detach().cpu().numpy().reshape(-1) # (bs, 1) => (bs)
                metrics["preds"] = preds # (bs)

                labels = np.array(utils.get_from_mapping(input, 'label')) # (bs)
                metrics["gts"] = labels
            else:
                preds = preds.detach().cpu().numpy().reshape(-1) # (bs, 1) => (bs)
                metrics["preds"] = np.concatenate((metrics["preds"], preds))
                labels = np.array(utils.get_from_mapping(input, 'label')) # (bs)
                metrics["gts"] = np.concatenate((metrics["gts"], labels))

        return metrics
    
    def set_output_dir(self, path):
        self.output_dir = path

    def display(self, metrics, epoch, step=None, lr=None, time=None):
        if lr is not None:
            print("Epoch = {}, Step = {}".format(epoch, step))
        else:
            print("************************* Validation *************************")
        
        if 'reg_loss' not in metrics:
            print(metrics.keys())

        loss = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        if lr is not None:
            print("loss = %2.4f, time = %2.4f" % (loss, time))
        else:
            rvalue, pvalue, rrmse = 0, 0, 0
            if "preds" in metrics and "gts" in metrics:
                preds = metrics["preds"]
                gts = metrics["gts"]
                slope,intercept,rvalue,pvalue,stderr = linregress(gts, preds)
                # rvalue 表示 皮尔森系数，越接近1越好，一般要到0.75以上，预测合格，pvalue表示检验的p值，需要小于0.05，严格一点需要小于0.01
                rrmse = sklearn_metrics.mean_squared_error(gts, preds)
                # rrmse 表示实验值和预测值之间的均方根误差，值越接近于0越好
                save_dir = os.path.join(self.output_dir, "prediction")
                if not os.path.exists(save_dir):
                    print("Directory {} doesn't exist, create a new.".format(save_dir))
                    os.makedirs(save_dir)
                    
                output_file = os.path.join(save_dir, "pred_output_{}".format(epoch))
                np.savez(output_file, preds=preds, gts=gts)

                info = {'loss': loss, 'rvalue': rvalue, 'pvalue': pvalue, 'rrmse': rrmse}
                output_file = os.path.join(save_dir, "info_{}_{:.3f}".format(epoch, rvalue))
                np.savez(output_file, info=info)


            print("validation loss = %2.4f, rvalue = %2.4f, pvalue = %2.8f, rrmse = %2.4f" % (loss, rvalue, pvalue, rrmse))
            # print("gts: {} ".format(gts))
            # print("preds: {} ".format(preds))