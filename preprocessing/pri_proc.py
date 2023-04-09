import pandas as pd
import os
import numpy as np
from preprocessing.protein_chemistry import list_aa, \
  aa_to_index, dictionary_covalent_bonds

import torch
from transformers import T5EncoderModel, T5Tokenizer









if __name__ == '__name__':
  data_folder = '../data/'
  seq_data = 'seq_dg_identity_230404.txt'
  df = pd.read_csv(os.path.join(data_folder, seq_data), sep='\t')

  all_seq_by_str = [] # list of list of str
  all_seq_by_int = [] # list of list of int
  for r in range(len(df)):
    # sequence
    prot_seq = df.loc[r]['protein_sequence']
    seq_by_int = [aa_to_index[it] # list_aa.index(it) 
      if it in list_aa else it.upper() # handle lower case of aa
      for it in prot_seq 
      if it in list_aa or it.upper() in list_aa] # check if aa not in list_aa
    all_seq_by_int.append(seq_by_int)
    all_seq_by_str.append([it_aa for it_aa in prot_seq])
    # atoms
    for it in seq_by_int:
      aa = list_aa[it]
      atoms = dictionary_covalent_bonds[aa] # dict of atoms
      # we could get atom's feature from outside
      aa_atoms = list(atoms.keys())


  #
  use_embedding = False
  if use_embedding:
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")  

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

  #









