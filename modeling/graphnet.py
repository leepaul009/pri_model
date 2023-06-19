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
            [batch_size, max_vector_num, max_vector_num], device=device) # atom case: [aa, max_atoms, max_atoms]
        
        if self.config['use_atom_embedding']:
            # [max_atoms,1] => [max_atoms,12]
            if hidden_states.type is not torch.long:
                hidden_states = hidden_states.type(torch.long).squeeze(-1)
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
        # TODO: for tensor = (bs*max_seq, max_atoms, h), we should max pool along true atoms dim
        '''
        # suppose we have list_real_atoms_num
        list_feat_per_aa = []
        for i in range(hidden_states.shape[0]):
            iend = list_real_atoms_num[i]
            feat_per_aa = torch.max(hidden_states[i, :iend], dim=0)[0] # (max_atoms, h) => (real_atoms, h) => (h)
            list_feat_per_aa.append(feat_per_aa)
        final_feat = torch.cat(list_feat_per_aa)
        return final_feat, ... # (bs*max_seq, h)
        '''
        return torch.max(hidden_states, dim=1)[0], \
               torch.cat(utils.de_merge_tensors(hidden_states, lengths))

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
        # w=seq,h=4 ==> output: w'=seq-2(seq-3+1), h'=1(4-4+1)
        # h-kh+1:, w-kw+1:
        if self.use_conv:
            self.na_conv2d = nn.Conv2d(in_channels=1, out_channels=hidden_size // 2, kernel_size=(3,4))
        
        self.use_kmers = True
        if self.use_kmers:
            out_channels = hidden_size // 2
            if self.use_chemistry:
                out_channels = hidden_size // 4
            self.kmers_net = KMersNet(base_channel=8, out_channels=out_channels)

        if not self.use_kmers and not self.use_conv:
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

    def forward_encode_sub_graph(
            self, 
            mapping: List[Dict],  # not used
            atom_attributes, # List[List[np.ndarray]]
            atom_indices, # not used
            na_embedding,
            # matrix: List[np.ndarray], 
            # polyline_spans: List[List[slice]],
            device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
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
            a, b = self.atom_level_sub_graph(prot_atom_input_list_list[i])
            prot_states_batch.append(a)
        
        # TODO: combine bs*seq dim
        '''
        
        prot_lengths = []
        resi_lengths = []
        for i in range(batch_size):
            num_aa = len(atom_attributes[i])
            prot_lengths.append(num_aa)
            for j in range(len(atom_attributes[i])): # per aa
                num_atom = len(atom_attributes[i][j])
                resi_lengths.append(num_atom)
        temp_tensor = torch.zeros([batch_size, max(prot_lengths), max(resi_lengths), 1], device=device)
        for i in range(batch_size): # per chain
            for j in range(len(atom_attributes[i])): # per aa
                # np.array to tensor shape=[num_atoms, 1]
                tensor = torch.tensor(atom_attributes[i][j], device=device) # (atoms, 1)
                temp_tensor[i][j][:tensor.shape[0]] = tensor
        temp_tensor = temp_tensor.reshape(batch_size*max(prot_lengths), max(resi_lengths), 1)
        prot_hidden, _ = self.atom_level_sub_graph(temp_tensor)  # (bs*max_seq, h)
        prot_hidden = prot_hidden.reshape(batch_size, max(prot_lengths), -1)
        '''

        # TODO: decide if we should use subgraph of rdna atoms
        rdna_states_batch = []
        for i in range(len(drna_atom_input_list_list)): # per chain
            a, b = self.atom_level_sub_graph(drna_atom_input_list_list[i])
            rdna_states_batch.append(a)

        # TODO:
        '''
        # (bs, max_seq, h) => list[ (real_seq, h) ]
        
        '''

        for i in range(batch_size):
            prot_feats = prot_states_batch[i] # [aa, h/2]
            if len(rdna_states_batch) != 0:
                rdna_feats = rdna_states_batch[i] # [nc, h/2]
            else:
                # rdna_feats = prot_feats
                rdna_feats = na_embedding[i] # (n_nc, h/2)
            rdna_feats = rdna_feats + self.laneGCN_A2L(
                rdna_feats.unsqueeze(0), prot_feats.unsqueeze(0)).squeeze(0)
            rdna_feats = rdna_feats + self.laneGCN_L2L(
                rdna_feats.unsqueeze(0)).squeeze(0)
            prot_feats = prot_feats + self.laneGCN_L2A(
                prot_feats.unsqueeze(0), rdna_feats.unsqueeze(0)).squeeze(0)
            
            if self.use_A2A:
                prot_feats = prot_feats + self.laneGCN_A2A(
                    prot_feats.unsqueeze(0)).squeeze(0)

            prot_states_batch[i] = torch.cat([prot_feats, rdna_feats]) # [aa+nc, h/2]
            # prot_states_batch[i] = prot_feats # [aa, h/2]

        return prot_states_batch, rdna_states_batch
    
    def aa_attribute_embeding(
            self,
            aa_attributes : List[np.ndarray], 
            aa_pwm : List[np.ndarray],
            # aa_indices,
            device, batch_size) -> Tuple[List[Tensor], List[int]]:
        """
        compute embedding of amino acid
        inputs:
            aa_attributes: List[Tensor=(n_aa, 20)]
        outputs:
            List[Tensor=(n_aa, h/2)]
            
        """
        input_list = []
        for i in range(batch_size):
            tensor = torch.tensor(aa_attributes[i], device=device)
            input_list.append(tensor)
        aa_embedding, lengths = utils.merge_tensors(input_list, device=device) # [bs, max_n_aa, 20]
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

        return utils.de_merge_tensors(final_embedding, lengths), lengths
        # return aa_embedding, lengths

    def na_attribute_embeding(
            self,
            na_attributes : List[np.ndarray], 
            na_other_attributes : List[np.ndarray], 
            # aa_indices,
            device, batch_size) -> Tuple[List[Tensor], List[int]]:
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

        return utils.de_merge_tensors(na_embedding, lengths), lengths
        # return na_embedding, lengths

    # @profile
    def forward(self, mapping: List[Dict], device):

        import time
        global starttime
        starttime = time.time()

        # amino acid feature, tensor shape = (number_of_amino_acid, 20)
        aa_attributes = utils.get_from_mapping(mapping, 'aa_attributes') # List[np.ndarray=(n_aa, 20)]

        aa_pwm = None
        if self.pwm_type == 'hmm':
            aa_pwm = utils.get_from_mapping(mapping, 'aa_hmm_pwm') # List[np.ndarray=(n_aa, 30)]
        elif self.pwm_type == 'pssm':
            aa_pwm = utils.get_from_mapping(mapping, 'aa_pssm_pwm') # List[np.ndarray=(n_aa, 30)]
        elif self.pwm_type == 'psfm':
            aa_pwm = utils.get_from_mapping(mapping, 'aa_psfm_pwm') # List[np.ndarray=(n_aa, 30)]
        else:
            NotImplemented

        aa_indices = utils.get_from_mapping(mapping, 'aa_indices') # not used
        atom_attributes = utils.get_from_mapping(mapping, 'atom_attributes') # List[List[np.ndarray=(n_atoms,)]]
        atom_indices = utils.get_from_mapping(mapping, 'atom_indices')
        # DNA/RNA feature, tensor shape = (number_of_DNA/RNA, 4)
        na_attributes = utils.get_from_mapping(mapping, 'nucleotide_attributes') # List[np.ndarray=(n_nc, 4)]

        if self.use_chemistry:
            na_other_attributes = utils.get_from_mapping(mapping, 'nucleotide_other_attributes') # List[np.ndarray=(n_nc, 10)]
        else:
            na_other_attributes = None

        batch_size = len(aa_attributes)

        # compute embedding feature for amino acid sequence
        # list[Tensor=(n_aa, h/2)]
        aa_embedding, _ = self.aa_attribute_embeding(aa_attributes, aa_pwm, device, batch_size)
        
        # compute embedding feature for DNA/RNA
        # list[Tensor=(n_nc, h/2)]
        na_embedding, _ = self.na_attribute_embeding(na_attributes, na_other_attributes, device, batch_size)
        
        # [bs, max(n_aa+n_nc), h/2]
        element_states_batch, lane_states_batch =\
            self.forward_encode_sub_graph(
                mapping, 
                atom_attributes, atom_indices, na_embedding,
                device, batch_size)
        

        # [bs, max(n_aa+n_nc), h/2], inputs_lengths: list of actual len_aa+len_nc of each batch
        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)

        # create mask
        max_seq_num = max(inputs_lengths)
        # (bs, max_n_aa, max_n_aa)
        attention_mask = torch.zeros([batch_size, max_seq_num, max_seq_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)
        
        # TODO: try different method to aggregate features rather than concat
        complex_embeddings = []
        seq_len_per_complex = []
        for i in range(batch_size):
            seq_len_per_complex.append((aa_embedding[i].shape[0], na_embedding[i].shape[0]))
            complex_emb = torch.cat([aa_embedding[i], na_embedding[i]]) # (n_aa+n_nc, h/2)
            complex_embeddings.append(complex_emb)
        # [bs, max(n_aa+n_nc), h/2], inputs_lengths: list of actual len_aa+len_nc of each batch
        complex_embeddings, complex_lengths = utils.merge_tensors(complex_embeddings, device=device)

        # [bs, max(n_aa+n_nc), h/2], [bs, max(n_aa+n_nc), h/2]=> [bs, max(n_aa+n_nc), h]
        inputs = torch.cat([complex_embeddings, inputs], dim=-1) 

        # global_graph: GlobalGraphRes [bs, max(n_aa), h]
        hidden_states = self.global_graph(inputs, attention_mask, mapping)
        
        # to list[complex_feat], list number = batch size
        hidden_state_per_complex = utils.de_merge_tensors(hidden_states, complex_lengths)
        combined_feat_per_complex = []
        for i in range(batch_size):
            len_aa, len_nc = seq_len_per_complex[i]
            assert(complex_lengths[i] == len_aa + len_nc)
            seq_feat_aa = hidden_state_per_complex[i][:len_aa] # (n_aa, h)
            seq_feat_nc = hidden_state_per_complex[i][len_aa:] # (n_nc, h)
            unit_feat_aa = torch.max(seq_feat_aa, dim=0)[0] # (h,)
            unit_feat_nc = torch.max(seq_feat_nc, dim=0)[0] # (h,)
            combined_feat_per_complex.append( torch.cat([unit_feat_aa, unit_feat_nc]).unsqueeze(0) ) # (1, h*2)
        hidden_states = torch.cat(combined_feat_per_complex) # => (bs, h*2)

        # [bs, max(n_aa), h] => [bs, h]
        # hidden_states = torch.max(hidden_states, dim=1)[0]

        # TODO: fine tune the structure of predicting score
        
        hidden_states = self.reg_head(hidden_states) # (bs, h*2)=>(bs, h)
        outputs = self.reg_pred(hidden_states) # [bs, h] => [bs, 1]
         
        labels = utils.get_from_mapping(mapping, 'label')
        labels = torch.tensor(labels, device=device, dtype=torch.float32).reshape(-1, 1) # [bs, 1]

        loss_output = self.loss(outputs, labels)

        return loss_output['loss'], outputs, loss_output


class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

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
    


    def display(self, metrics, epoch, step=None, lr=None):
        if lr is not None:
            print("Epoch = {}, Step = {}".format(epoch, step))
        else:
            print("************************* Validation *************************")
        loss = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        if lr is not None:
            print("loss = %2.4f" % (loss))
        else:
            rvalue, pvalue, rrmse = 0, 0, 0
            if "preds" in metrics and "gts" in metrics:
                preds = metrics["preds"]
                gts = metrics["gts"]
                slope,intercept,rvalue,pvalue,stderr = linregress(gts, preds)
                # rvalue 表示 皮尔森系数，越接近1越好，一般要到0.75以上，预测合格，pvalue表示检验的p值，需要小于0.05，严格一点需要小于0.01
                rrmse = sklearn_metrics.mean_squared_error(gts, preds)
                # rrmse 表示实验值和预测值之间的均方根误差，值越接近于0越好
                np.savez("pred_output_{}".format(epoch), preds=preds, gts=gts)


            print("validation loss = %2.4f, rvalue = %2.4f, pvalue = %2.8f, rrmse = %2.4f" % (loss, rvalue, pvalue, rrmse))
            print("gts: {} ".format(gts))
            print("preds: {} ".format(preds))