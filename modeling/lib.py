import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

# import utils

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(self.module.weight, a=1)
            if self.module.bias is not None:
                nn.init.constant_(self.module.bias, 0)

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class Dense(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True, activation=None, device=None, dtype=None):
        super(Dense, self).__init__()
        self.activation = activation
        self.linear = torch.nn.Linear(in_features, out_features, bias=use_bias)
        
        nn.init.kaiming_uniform_(self.linear.weight, a=1)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        if self.activation == 'relu':
            hidden_states = torch.nn.functional.relu(hidden_states)
        else: # TODO: apply other activation function
            NotImplemented
        return hidden_states


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)
        # self.layer_norm = nn.BatchNorm1d(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        
        if isinstance(self.layer_norm, nn.BatchNorm1d):
            orig_shape = hidden_states.shape
            hidden_states = self.layer_norm(hidden_states.reshape(-1, orig_shape[-1]))
            hidden_states = hidden_states.reshape(orig_shape)
        else:
            hidden_states = self.layer_norm(hidden_states)
        
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class GlobalGraph(nn.Module):
    r"""
    Global graph
    It's actually a self-attention.
    """

    # hidden_size = attention_head_size * num_attention_heads
    def __init__(self, 
                 hidden_size, 
                 attention_head_size=None, 
                 num_attention_heads=1, 
                 use_attention_decay=False):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads \
            if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

        self.use_attention_decay = use_attention_decay
        # if utils.args.attention_decay:
        if self.use_attention_decay:
            self.attention_decay = nn.Parameter(torch.ones(1) * 0.5)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.
        After this function, 1 turns to 0, 0 turns to -10000.0
        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1) # [bs, 1, seq, seq]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)
    
    """
    inputs:
        hidden_states: Tensor=(bs, seq, hidden)
            case1: bs=batch_size, seq=num_aa
            case2: bs=num_aa, seq=num_atoms_in_curr_aa
        attention_mask: Tensor=(bs, seq, seq)
    outputs:
        Tensor=(bs, seq, hidden), dimension same to input hidden_states
    """
    def forward(self, 
                hidden_states, 
                attention_mask=None, 
                mapping=None, 
                return_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # shape=[bs, heads, seq_query, seq_key]
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        # if utils.args.attention_decay and utils.second_span:
        #     attention_scores[:, 0, 0, 0] = attention_scores[:, 0, 0, 0] - self.attention_decay
        # mask item is -10000 for attention_scores, softmax return 0
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # if utils.args.visualize and mapping is not None:
        #     for i, each in enumerate(attention_probs.tolist()):
        #         mapping[i]['attention_scores'] = np.array(each[0])
        # if utils.args.attention_decay and utils.second_span:
        if self.use_attention_decay:
            # utils.logging(self.attention_decay, prob=0.01)
            value_layer = torch.cat([value_layer[:, 0:1, 0:1, :] * self.attention_decay,
                                     value_layer[:, 0:1, 1:, :]],
                                    dim=2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            assert attention_probs.shape[1] == 1
            attention_probs = torch.squeeze(attention_probs, dim=1)
            assert len(attention_probs.shape) == 3
            return context_layer, attention_probs
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, 
                 hidden_size, 
                 attention_head_size=None, 
                 num_attention_heads=1, 
                 key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, 
                hidden_states_query, 
                hidden_states_key=None, 
                attention_mask=None, 
                mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (batch, head, max_vector_num, head_size), head维度不参与att
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer) # [batch, head, query_dim, feat]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,) # all_head_size = head * feat
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        # internal dim = hidden/2
        # self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        # self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)
        
        # self.global_graph = GlobalGraph(hidden_size, hidden_size)
        # self.global_graph2 = GlobalGraph(hidden_size, hidden_size)
        
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) 
                                for _ in range(3)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(3)])
        
        self.use_dropout = True
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        # hidden_states = self.global_graph(hidden_states, attention_mask, mapping) \
        #                 + self.global_graph2(hidden_states, attention_mask, mapping)
        # hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask, mapping),
        #                            self.global_graph2(hidden_states, attention_mask, mapping)], dim=-1)
        
        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask, mapping)
            if self.use_dropout:
                hidden_states = self.dropout(hidden_states)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        # if self.use_dropout:
        #     hidden_states = self.dropout(hidden_states)
        return hidden_states


class PointSubGraph(nn.Module):
    """
    Encode 2D goals conditioned on target agent
    """

    def __init__(self, hidden_size):
        super(PointSubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MLP(2, hidden_size // 2),
                                     MLP(hidden_size, hidden_size // 2),
                                     MLP(hidden_size, hidden_size)])

    def forward(self, hidden_states: Tensor, agent: Tensor):
        device = hidden_states.device
        predict_agent_num, point_num = hidden_states.shape[0], hidden_states.shape[1]
        hidden_size = self.hidden_size
        assert (agent.shape[0], agent.shape[1]) == (predict_agent_num, hidden_size)
        agent = agent[:, :hidden_size // 2].unsqueeze(1).expand([predict_agent_num, point_num, hidden_size // 2])
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(torch.cat([hidden_states, agent], dim=-1))

        return hidden_states
