import itertools
import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
import shutil
import collections
import torch
from pathlib import Path

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}

VOCAB_KMER = {
  "69": "3",
  "261": "4",
  "1029": "5",
  "4101": "6",}
  
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class Alphabet(object):

  SPECIAL_TOKENS_ATTRIBUTES = [
      "bos_token",
      "eos_token",
      "unk_token",
      "sep_token",
      "pad_token",
      "cls_token",
      "mask_token",
      "additional_special_tokens",
  ]

  def __init__(
      self,
      vocab_file,
      standard_toks: Sequence[str],
      prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
      append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
      prepend_bos: bool = True, #T
      append_eos: bool = False, #T
      use_msa: bool = False, #F
  ):
    self.standard_toks = list(standard_toks)
    self.prepend_toks = list(prepend_toks) # ("<pad>", "<unk>", "<cls>", "<sep>")
    self.append_toks = list(append_toks) # ("<mask>",)
    self.prepend_bos = prepend_bos
    self.append_eos = append_eos
    self.use_msa = use_msa

    self.max_length = 512 # TODO
    self.vocab = load_vocab(vocab_file)
    self.kmer = VOCAB_KMER[str(len(self.vocab))] # 6
    # self.all_toks = list(self.prepend_toks)
    # self.all_toks.extend(self.standard_toks)
    # for i in range((8 - (len(self.all_toks) % 8)) % 8):
    #   self.all_toks.append(f"<null_{i  + 1}>")
    # self.all_toks.extend(self.append_toks)
    self.all_toks = [tok for tok, _ in self.vocab.items()]

    # self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
    self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
    self.tok_to_idx = {tok: i for tok, i in self.vocab.items()}

    self.padding_idx = self.tok_to_idx["<pad>"]
    self.unk_idx = self.tok_to_idx["<unk>"]
    self.cls_idx = self.tok_to_idx["<cls>"]
    self.sep_idx = self.tok_to_idx["<sep>"]
    self.mask_idx = self.tok_to_idx["<mask>"]
    
    # self.eos_idx = self.get_idx("<eos>")
    self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>', "<sep>"]
    self.unique_no_split_tokens = self.all_toks # store all toks

  def __len__(self):
      return len(self.all_toks)

  # def get_idx(self, tok):
  #     return self.tok_to_idx.get(tok, self.unk_idx)

  def get_tok(self, ind):
      return self.all_toks[ind]

  def to_dict(self):
      return self.tok_to_idx.copy()

  def get_batch_converter(self, truncation_seq_length: int = None):
      # if self.use_msa:
      #     return MSABatchConverter(self, truncation_seq_length)
      # else:
          return BatchConverter(self, truncation_seq_length)

  @classmethod
  def from_architecture(cls, vocab_file: str) -> "Alphabet":
      standard_toks = proteinseq_toks["toks"]
      prepend_toks = ("<pad>", "<unk>", "<cls>", "<sep>")
      append_toks = ("<mask>",)
      prepend_bos = True
      append_eos = True
      use_msa = False
      # vocab_file = 'dataset/checkpoints/dvocab.txt'
      return cls(vocab_file, standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

  def _tokenize(self, text) -> List[str]:
      return text.split()

  def tokenize(self, text, **kwargs) -> List[str]:

      def split_on_token(tok, text):
          result = []
          split_text = text.split(tok)
          for i, sub_text in enumerate(split_text):
              sub_text = sub_text.rstrip()
              if i == 0 and not sub_text:
                  result += [tok]
              elif i == len(split_text) - 1:
                  if sub_text:
                      result += [sub_text]
                  else:
                      pass
              else:
                  if sub_text:
                      result += [sub_text]
                  result += [tok]
          return result

      def split_on_tokens(tok_list, text):
          if not text.strip(): #F
              return []
          if not tok_list: #F
              return self._tokenize(text)

          tokenized_text = []
          text_list = [text]
          for tok in tok_list:
              tokenized_text = []
              for sub_text in text_list:
                  if sub_text not in self.all_special_tokens:
                      tokenized_text += split_on_token(tok, sub_text)
                  else:
                      tokenized_text += [sub_text]
              text_list = tokenized_text

          return list(
              itertools.chain.from_iterable(
                  (
                      self._tokenize(token) 
                      if token not in self.all_special_tokens 
                      else [token]
                      for token in tokenized_text
                  )
              )
          )

      no_split_token = self.all_special_tokens # all toks
    
      ### TODO: fix logic that "convert U to T"
      text = text.replace("U", "T")
      tokenized_text = split_on_tokens(no_split_token, text)
      
      return tokenized_text
  
  def encode(self, text):
      # return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
      if isinstance(text, (list, tuple)): # double seqs
          text_0, text_1 = text
          ids_0 = [self.tok_to_idx[tok] for tok in self.tokenize(text_0)]
          ids_1 = [self.tok_to_idx[tok] for tok in self.tokenize(text_1)]

          total_len = len(ids_0) + len(ids_1) + self.num_added_tokens(pair=True)
          # TODO: total_len > self.max_length truncate_sequences

          sequence = self.build_inputs_with_special_tokens(ids_0, ids_1)
          token_type_ids = self.create_token_type_ids_from_sequences(ids_0, ids_1)

      else: # single seq
          ids = [self.tok_to_idx[tok] for tok in self.tokenize(text)]
          total_len = len(ids) + self.num_added_tokens(pair=False)
          # TODO: total_len > self.max_length truncate_sequences

          sequence = self.build_inputs_with_special_tokens(ids)
          token_type_ids = self.create_token_type_ids_from_sequences(ids)

      encoded_inputs = {}
      encoded_inputs["input_ids"] = sequence
      encoded_inputs["token_type_ids"] = token_type_ids
      encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
      return encoded_inputs

  def num_added_tokens(self, pair=False):
      token_ids_0 = []
      token_ids_1 = []
      return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
      cls = [self.cls_idx] # CLS 2
      sep = [self.sep_idx] # SEP 3

      if token_ids_1 is None: # case of single sequence
        if len(token_ids_0) < 510:
            return cls + token_ids_0 + sep # [2, 3]
        else:
            output = []
            num_pieces = int(len(token_ids_0)//510) + 1 # 1+1
            for i in range(num_pieces):
                # i=0, 0:510
                # i=1, 510:len, len >=510
                output.extend(cls + token_ids_0[ 510*i:min(len(token_ids_0), 510*(i+1)) ] + sep)
            return output
      # case of double sequences
      return cls + token_ids_0 + sep + token_ids_1 + sep

  def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
      """
      Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
      A BERT sequence pair mask has the following format:
      0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
      | first sequence    | second sequence

      if token_ids_1 is None, only returns the first portion of the mask (0's).
      """
      sep = [self.sep_idx]
      cls = [self.cls_idx]
      if token_ids_1 is None:
          if len(token_ids_0) < 510:
              return len(cls + token_ids_0 + sep) * [0] # ([CLS] A [SEP])x0
          else:
              num_pieces = int(len(token_ids_0)//510) + 1
              return (len(cls + token_ids_0 + sep) + 2*(num_pieces-1)) * [0]
      return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]


  def get_special_tokens_mask(self, token_ids_0) -> List[int]:
    """
    Retrieves sequence ids from a token list that has no special tokens added.
    Returns:
        A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    """
    return list(map(lambda x: 1 if x in [self.sep_idx, self.cls_idx, self.padding_idx] else 0, token_ids_0))



class BatchConverter(object):
    """Callable to convert an unprocessed (strings) batch to a
    processed (tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        batch_size = len(raw_batch)
        
        seq_list = [it['seqs_kmers'] for it in raw_batch]
        # seq_list = [(s0, s1) for s0, s1 in raw_batch]
        
        # list of dict {input_ids, token_type_ids, attention_mask(useless)}
        seq_encoded_list = [self.alphabet.encode(seq) for seq in seq_list]
        
        if self.truncation_seq_length: # only keep certain length of seq_str
            # seq_encoded_list = [seq_str['input_ids'][:self.truncation_seq_length] 
            #                     for seq_str in seq_encoded_list]
            old_list = seq_encoded_list
            seq_encoded_list = []
            for seq_obj in old_list:
                seq_obj['input_ids']      = seq_obj['input_ids'][:self.truncation_seq_length]
                seq_obj['token_type_ids'] = seq_obj['token_type_ids'][:self.truncation_seq_length]
                seq_obj['attention_mask'] = seq_obj['attention_mask'][:self.truncation_seq_length]
                seq_encoded_list.append(seq_obj)
        
        max_len = max(len(seq_encoded['input_ids']) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                # what is prepend_bos and append_eos? beg and end position
                # max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
                max_len,
            ),
            dtype=torch.int64,
        ) # (B,T=max_T)
        tokens.fill_(self.alphabet.padding_idx) # padding index (<pad>'s id)

        for i, seq_encoded in enumerate(seq_encoded_list):
            input_ids = seq_encoded['input_ids']
            token_type_ids = seq_encoded['token_type_ids']

            # if self.alphabet.prepend_bos:
            #     tokens[i, 0] = self.alphabet.cls_idx # put cls_idx in pos_0 
            seq = torch.tensor(input_ids, dtype=torch.int64)
            tokens[
                i,
                # int(self.alphabet.prepend_bos) : len(seq_encoded) + int(self.alphabet.prepend_bos),
                0 : len(seq)
            ] = seq
            # if self.alphabet.append_eos:
            #     # put eos_idx after last seq pos
            #     tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        # ex. after processed [ cls_idx, s0, s1, .... sN-1, eos_idx, pad_idx, pad_idx ]
        
        # return labels, strs, tokens
        # output = [each for each in raw_batch]
        return tokens
