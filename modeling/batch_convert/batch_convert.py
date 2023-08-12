import os
from typing import Sequence, Tuple, List, Union, Dict




class BasicBatchConvert(object):
  def __init__(self, alphabet, dalphabet):
    self.alphabet  = alphabet
    self.dalphabet = dalphabet
    self.aa_convert = alphabet.get_batch_converter()
    self.nc_convert = dalphabet.get_batch_converter()
    
  def __call__(self, raw_batch: Sequence[Dict]):
    
    batch_size = len(raw_batch)
    aa_tokens = self.aa_convert(raw_batch)
    nc_tokens = self.nc_convert(raw_batch)
    return raw_batch, aa_tokens, nc_tokens