# import copy
# import json
# import logging
import os
from typing import Dict, Optional, Tuple




class DbmConfig(object):
    def __init__(
        self,
        
        # # architectures = ["BertForMaskedLM"],
        # bos_token_id = 0,
        # do_sample = False,
        # eos_token_ids = 0,
        # finetuning_task = None,
        # # id2label = {0 = "LABEL_0", 1 = "LABEL_1"},
        # initializer_range = 0.02,
        # # label2id = {LABEL_0 = 0, LABEL_1 = 1 },
        # length_penalty = 1.0,
        # max_length = 20,
        # model_type = "bert",
        # num_beams = 1,
        # num_labels = 2,
        # num_return_sequences = 1,
        # num_rnn_layer = 1,
        # output_past = True,
        # pad_token_id = 0,
        # pruned_heads = {},
        # repetition_penalty = 1.0,
        # rnn = "lstm",
        # rnn_dropout = 0.0,
        # rnn_hidden = 768,
        # split = 10,
        # temperature = 1.0,
        # top_k = 50,
        # top_p = 1.0,
        # torchscript = False,
        # use_bfloat16 = False,
        
        vocab_size = 4101,
        hidden_size = 768,
        max_position_embeddings = 512,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.1,
        #
        num_attention_heads = 12,
        output_attentions = False,
        attention_probs_dropout_prob = 0.1,
        #
        intermediate_size = 3072,
        hidden_act = "gelu",
        #
        is_decoder = False,
        output_hidden_states = False,
        num_hidden_layers = 12,
        
        # vocab_size=30522,
        # hidden_size=768,
        # num_hidden_layers=12,
        # num_attention_heads=12,
        # intermediate_size=3072,
        # hidden_act="gelu",
        # hidden_dropout_prob=0.1,
        # attention_probs_dropout_prob=0.1,
        # max_position_embeddings=512,
        # type_vocab_size=2,
        # initializer_range=0.02,
        # layer_norm_eps=1e-12,
        # split=10,
        # num_rnn_layer=1,
        # rnn_dropout=0.0,
        # rnn_hidden=768,
        # rnn="lstm",
        **kwargs
    ):
        # super().__init__(**kwargs)
        # self.vocab_size = vocab_size
        # self.hidden_size = hidden_size
        # self.num_hidden_layers = num_hidden_layers
        # self.num_attention_heads = num_attention_heads
        # self.hidden_act = hidden_act
        # self.intermediate_size = intermediate_size
        # self.hidden_dropout_prob = hidden_dropout_prob
        # self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.max_position_embeddings = max_position_embeddings
        # self.type_vocab_size = type_vocab_size
        # self.initializer_range = initializer_range
        # self.layer_norm_eps = layer_norm_eps
        # self.split = split
        # self.num_rnn_layer = num_rnn_layer
        # self.rnn = rnn
        # self.rnn_dropout = rnn_dropout
        # self.rnn_hidden = rnn_hidden
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        #
        self.num_attention_heads = num_attention_heads
        self.output_attentions = output_attentions
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        #
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        #
        self.is_decoder = is_decoder
        self.output_hidden_states = output_hidden_states
        self.num_hidden_layers = num_hidden_layers