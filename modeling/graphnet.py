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
from modeling.sub_graphnet import SubGraph
from modeling.input_embedding import ProtEmbedding, NcEmbedding, \
                                     ProtDeepEmbedding, NcDeepEmbedding

from preprocessing.protein_chemistry import list_atoms, max_num_atoms_in_aa, max_num_prot_chm_feats, \
    pwm_embeding_dims_by_type, is_pwm_type_valid

from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics

# post_preprocessor




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
        ### update output dim
        aa_emb_hidden_dim = sub_graph_hidden_dim if self.use_sub_graph else hidden_size
        self.pwm_type = args.pwm_type
        self.use_prot_chm = args.use_prot_chm_feature
        if not args.use_deep_emb:
          self.prot_layer = ProtEmbedding(args, aa_emb_hidden_dim)
        else:
          self.prot_layer = ProtDeepEmbedding(args, aa_emb_hidden_dim)

        ##################################################################
        ### nc embedding
        ##################################################################
        nc_emb_hidden_dim = sub_graph_hidden_dim if self.use_sub_graph else hidden_size
        self.use_nc_chm = args.use_chemistry
        if not args.use_deep_emb:
          self.nc_layer = NcEmbedding(args, nc_emb_hidden_dim)
        else:
          self.nc_layer = NcDeepEmbedding(args, nc_emb_hidden_dim)
        

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
        aa_embedding, prot_lengths = self.prot_layer(aa_attributes, aa_pwm, aa_chm, device, batch_size)
        # compute embedding feature for DNA/RNA, (N, max(num_nc), h/2)
        na_embedding, nc_lengths = self.nc_layer(na_attributes, na_other_attributes, device, batch_size)
        
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


