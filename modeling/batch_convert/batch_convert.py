import os
from typing import Sequence, Tuple, List, Union, Dict
import torch



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


    # feat_dim = raw_batch[0]['aa_chm'].shape[-1]
    aa_chm = torch.zeros((batch_size, 
                          aa_tokens.shape[-1], 
                          raw_batch[0]['aa_chm'].shape[-1]
                        ), dtype=torch.float32)
    for i, it in enumerate(raw_batch):
        aa_chm[i, 1:it['aa_chm'].shape[0]+1] = torch.tensor(it['aa_chm'])

    return raw_batch, aa_tokens, nc_tokens, aa_chm