from typing import Sequence, Tuple, List, Union
import copy
import math
import multiprocessing
import os
import pickle
import random
import zlib
from collections import defaultdict
from multiprocessing import Process
from random import choice
import argparse

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

import utils
from preprocessing.protein_chemistry import list_aa, \
  aa_to_index, dictionary_covalent_bonds, list_atoms, atom_type_mass, \
  nucleotide_to_index, max_num_atoms_in_aa, max_num_prot_chm_feats, \
  pwm_embeding_dims_by_type, is_pwm_type_valid
  
from data.dataset_process import processBinLabel, computeBin, getProtChemFeature, getProtChemFeature_7s

####################################

def remove_nan(matrix,padding_value=0.):
  aa_has_nan = np.isnan(matrix).reshape([len(matrix),-1]).max(-1)
  matrix[aa_has_nan] = padding_value
  return matrix

def binarize_categorical(matrix : np.ndarray, 
                         n_classes : int, 
                         out=None) -> np.ndarray:
  """
  Return one-hot feature map.

  Arguments:
  - matrix: np.ndarray, shape=(seq_num,)
  - n_classes: number of classes
  
  Returns:
  np.ndarray, shape=(seq_num, n_classes)
  """
  L = matrix.shape[0]
  matrix = matrix.astype(np.int32)
  if out is None:
    out = np.zeros([L, n_classes], dtype=np.bool_)
  subset = (matrix>=0) & (matrix<n_classes)
  out[np.arange(L)[subset],matrix[subset]] = 1
  return out

def process_sequence_without_coordinates(chain):
  chain_by_int = [aa_to_index[it] # list_aa.index(it) 
    if it in list_aa else aa_to_index[it.upper()] # handle lower case of aa
    for it in chain 
    if it in list_aa or it.upper() in list_aa] # check if aa not in list_aa
  return np.array(chain_by_int)

def get_pwm_feature(type, pwm_dir, input_complex):

  protein_index = input_complex['protein_index']

  data_file = os.path.join(pwm_dir, protein_index, "pssm_hmm.txt")
  pwm_df = pd.read_csv(data_file, sep='\t')
  # hmm
  hmm_cols = pwm_df.columns[pwm_df.columns.str.startswith("hmm")]
  num_hmm_cols = len(hmm_cols)
  hmm_np_array = pwm_df.iloc[:, 1:1+num_hmm_cols].values # (num_aa, 30)
  # pssm
  pssm_cols = pwm_df.columns[pwm_df.columns.str.startswith("pssm")]
  num_pssm_cols = len(pssm_cols)
  ibeg = 1+num_hmm_cols
  iend = 1+num_hmm_cols+num_pssm_cols
  pssm_np_array = pwm_df.iloc[:, ibeg:iend].values
  # psfm
  psfm_cols = pwm_df.columns[pwm_df.columns.str.startswith("psfm")]
  num_psfm_cols = len(psfm_cols)
  ibeg = 1+num_hmm_cols+num_pssm_cols
  iend = 1+num_hmm_cols+num_pssm_cols+num_psfm_cols
  psfm_np_array = pwm_df.iloc[:, ibeg:iend].values

  # return hmm_np_array, pssm_np_array, psfm_np_array

  aa_pwm_feature = None
  if type == 'hmm':
    aa_pwm_feature = hmm_np_array # (num_aa, 30)
  elif type == 'pssm':
    aa_pwm_feature = pssm_np_array # (num_aa, 20)
  elif type == 'psfm':
    aa_pwm_feature = psfm_np_array # (num_aa, 20)
  elif type == 'hmm_pssm':
    aa_pwm_feature = np.concatenate([hmm_np_array, pssm_np_array], axis=-1) # (num_aa, 50)
  elif type == 'all':
    aa_pwm_feature = np.concatenate([hmm_np_array, pssm_np_array, psfm_np_array], axis=-1) # (num_aa, 70)
  
  return aa_pwm_feature

def get_nc_chemistry(na_jobid, input_nucleotide, fdir):
  nc_whole_length = len(input_nucleotide.replace('|', ''))
  list_temp_arr = []
  # might have one or two na_jobids
  sub_paths = na_jobid.split(',')
  for sub_path in sub_paths:
    f = None
    if 'DNA' in sub_path:
      f = os.path.join(fdir, sub_path, '{}_EIIP.txt'.format(sub_path))
    elif 'RNA' in sub_path: 
      f = os.path.join(fdir, sub_path, '{}_spotrna1d.txt'.format(sub_path))
    if os.path.exists(f):
      temp_df = pd.read_csv(f, sep='\t')
      temp_arr = temp_df.values[:,1:]
      if len(temp_arr.shape) == 1:
          temp_arr = temp_arr.reshape(len(temp_df), -1)
      list_temp_arr.append(temp_arr)
  # 
  final_arr = np.concatenate(list_temp_arr, axis=0)
  # create other features for DNA/RNA
  dna_cols = 1
  rna_cols = 9
  nc_feat_tensor = np.zeros((nc_whole_length, dna_cols + rna_cols), dtype=np.float32)
  if final_arr.shape[1] == 1:
    nc_feat_tensor[:, :dna_cols] = final_arr
  else:
    nc_feat_tensor[:, dna_cols:] = final_arr
  return nc_feat_tensor

def convert_to_conv_format(nucleotide_sequences: List[str], kmers: int = 5) -> List[str]:
  output_seqs = []
  for seq in nucleotide_sequences:
    n_out = len(seq) - kmers + 1
    convs = [seq[i:i+kmers] for i in range(n_out)]
    seq_convs = ' '.join(convs)
    output_seqs.append(seq_convs)
  
  return output_seqs[0] if len(output_seqs) == 1 else output_seqs

####################################

def pri_get_instance(input_complex, args, other_inputs):
  """
  Process one input.

  Arguments:
  - input_complex: one complex containing a amino acid sequence and nucleotide sequence

  Returns:
  map of input features and label.
  """
  input_protein = input_complex["protein_sequence"] # string
  input_nucleotide = input_complex["nucleotide_sequence"] # string
  label_dG = input_complex['dG'] if 'dG' in input_complex else input_complex['zscore']

  protein_sequences = process_sequence_without_coordinates(input_protein)
  num_aa = len(protein_sequences)

  # TODO: use bool instead.
  if not args.use_deep_emb:
    aa_attributes = binarize_categorical(
      protein_sequences, 20).astype(np.float32) # (seq_aa, 20)
  else:
    f = os.path.join('dataset/_embedding/prot', '{}.npz'
          .format(input_complex["protein_index"]))
    tmp = np.load(f, allow_pickle=True)
    aa_attributes = tmp['embedding']
  # aa_indices = np.array([i for i in range(num_aa)])

  atom_attributes = []
  atom_mass_attributes = []
  atom_indices = []
  aa_lengths = []
  for i, it in enumerate(protein_sequences): # per aa
    aa = list_aa[it] # get aa string
    aa_atoms = list(dictionary_covalent_bonds[aa].keys())
    num_atoms = len(aa_atoms)
    atom_type = np.array([list_atoms.index(x) for x in aa_atoms])
    atom_mass = np.array([atom_type_mass[x] for x in atom_type])
    # atom_type = atom_type.reshape(-1,1) # (num_atoms, 1)
    # atom_mass = atom_mass.reshape(-1,1) # (num_atoms, 1)
    atom_type = remove_nan(atom_type)
    atom_mass = remove_nan(atom_mass)
    atom_attributes.append(atom_type)
    atom_mass_attributes.append(atom_mass)
    # indicates each atom belong to the index of aa
    atom_indices.append(np.ones((num_atoms,), dtype=np.int32) * i)
    aa_lengths.append(num_atoms)

  feature_by_atom = np.zeros((num_aa, max_num_atoms_in_aa), dtype=np.float32)
  for i in range(num_aa):
    feature_by_atom[i, :aa_lengths[i]] = atom_attributes[i]

  atom_indices = np.concatenate(atom_indices, axis=0)
  atom_indices = remove_nan(atom_indices)
  aa_attributes = remove_nan(aa_attributes)
  

  ################# get pwm features #################
  pwm_dir = 'dataset/pwm_data/pwm' # TODO: move to param
  aa_pwm_feature = get_pwm_feature(args.pwm_type, pwm_dir, input_complex)

  ################# get chemistry features #################
  aa_chm_feature = None
  if args.use_prot_chm_feature:
    p_chemfeat_dir = 'dataset/protein_all_feas/'
    aa_chm_feature = getProtChemFeature(p_chemfeat_dir, input_complex, args) # (num_aa, 7)


  ################# get nucleotide features #################
  # nucleotide_to_index = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
  seqs_kmers = None
  nucleotide_sequences = None
  if '|' in input_nucleotide:
    nucleotide_sequences = input_nucleotide.split('|')
  else:
    nucleotide_sequences = [input_nucleotide]
  seqs_kmers = convert_to_conv_format(nucleotide_sequences, kmers=args.kmers)
  
  
  if not args.use_deep_emb:
    nucleotide_attributes = list()
    for seq in nucleotide_sequences:
      seq_data = np.array( [nucleotide_to_index[it] for it in seq] )
      # TODO: use bool instead.
      seq_data = binarize_categorical(seq_data, 4).astype(np.float32) # (num_nc, 4)
      # TODO: use other features
      # seq_data = np.concatenate([seq_data, other_feature], axis=1)
      seq_data = remove_nan(seq_data)
      nucleotide_attributes.append(seq_data)
    # TODO: how to use double chains? 
    #       combine to single chain or create 2 channel?
    nucleotide_attributes = np.concatenate(nucleotide_attributes)
  else:
    f = os.path.join('dataset/_embedding/nc', '{}.npz'
          .format(input_complex["key_nucleic_acids"]))
    tmp = np.load(f, allow_pickle=True)
    nucleotide_attributes = tmp['embedding']


  ############ use chemistry feature as input ############
  nc_chemistry_features = None
  if args.use_chemistry:
    chemistry_dir = 'dataset/nc_data' # TODO: move to param
    nc_chemistry_features = get_nc_chemistry(input_complex['na_jobid'], 
      input_complex["nucleotide_sequence"],  chemistry_dir)
  
  ### bin label
  bin_ctrs, bin_half_w, label_bin, label_offset = None, None, None, None
  if args.label_bin:
    # print("use bin label instead of simple regression")
    bin_ctrs, bin_half_w, label_bin, label_offset \
      = processBinLabel(args, label_dG)
  
  mapping = {}
  mapping.update(dict(
    exp_id = input_complex["exp_id"], # must
    protein = input_protein,
    nc_sequences = nucleotide_sequences,
    seqs_kmers = seqs_kmers,
    aa_attributes = aa_attributes,  # (num_aa, 20)
    aa_pwm = aa_pwm_feature,        # (num_aa, ?)
    aa_chm = aa_chm_feature,        # (num_aa, 37)
    # aa_indices = aa_indices, # (num_aa,)
    # atom_attributes = atom_attributes, # list[np.ndarray=(num_atoms,1)]
    atom_attributes = feature_by_atom, # (num_aa, MAX_NUM_ATOMS)
    aa_lengths = aa_lengths, # per aa length
    # atom_indices = atom_indices, # (all_atoms_in_aa, )
    nucleotide_attributes = nucleotide_attributes, # (num_nc', 4) num_nc' = sum(all chain)
    nucleotide_other_attributes = nc_chemistry_features, # (num_nc', 10) 
    label         = label_dG,     # float scalar
    bin_ctrs      = bin_ctrs,     # np.arr (n_bins, )
    bin_half_w    = bin_half_w,   # np.arr (n_bins, )
    label_bin     = label_bin,    # int scalar
    label_offset  = label_offset, # float scalar
  ))
  return mapping


def hox_get_instance(input_complex, args, other_inputs):
  input_protein    = input_complex["protein_sequence"] # string
  input_nucleotide = input_complex["nucleotide_sequence"] # string
  label_dG         = input_complex['dG'] if 'dG' in input_complex else input_complex['zscore']

  protein_sequences = process_sequence_without_coordinates(input_protein)
  num_aa = len(protein_sequences)

  # TODO: use bool instead.
  aa_attributes = binarize_categorical(
    protein_sequences, 20).astype(np.float32) # (seq_aa, 20)
  aa_indices = np.array([i for i in range(num_aa)])

  atom_attributes = []
  atom_mass_attributes = []
  atom_indices = []
  aa_lengths = []
  for i, it in enumerate(protein_sequences): # per aa
    aa = list_aa[it] # get aa string
    aa_atoms = list(dictionary_covalent_bonds[aa].keys())
    num_atoms = len(aa_atoms)
    atom_type = np.array([list_atoms.index(x) for x in aa_atoms])
    atom_mass = np.array([atom_type_mass[x] for x in atom_type])
    # atom_type = atom_type.reshape(-1,1) # (num_atoms, 1)
    # atom_mass = atom_mass.reshape(-1,1) # (num_atoms, 1)
    atom_type = remove_nan(atom_type)
    atom_mass = remove_nan(atom_mass)
    atom_attributes.append(atom_type)
    atom_mass_attributes.append(atom_mass)
    # indicates each atom belong to the index of aa
    atom_indices.append(np.ones((num_atoms,), dtype=np.int32) * i)
    aa_lengths.append(num_atoms)

  feature_by_atom = np.zeros((num_aa, max_num_atoms_in_aa), dtype=np.float32)
  for i in range(num_aa):
    feature_by_atom[i, :aa_lengths[i]] = atom_attributes[i]

  atom_indices = np.concatenate(atom_indices, axis=0)
  atom_indices = remove_nan(atom_indices)
  aa_attributes = remove_nan(aa_attributes)

  ################# get chemistry features #################

  prot_chm_df = other_inputs['prot_chm_df'] if 'prot_chm_df' in other_inputs else None
  aa_chm_feature = None
  if args.use_prot_chm_feature:
    assert prot_chm_df is not None
    p_chemfeat_dir = 'dataset/protein_all_feas/'
    aa_chm_feature = getProtChemFeature_7s(prot_chm_df, input_complex, args) # (num_aa, 7)


  ################# get nucleotide features #################
  # nucleotide_to_index = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
  seqs_kmers = None
  nucleotide_sequences = None
  if '|' in input_nucleotide:
    nucleotide_sequences = input_nucleotide.split('|')
  else:
    nucleotide_sequences = [input_nucleotide]
  
  seqs_kmers = convert_to_conv_format(nucleotide_sequences, 
                                      kmers=args.kmers)
   
  nucleotide_attributes = list()
  for seq in nucleotide_sequences:
    seq_data = np.array( [nucleotide_to_index[it] for it in seq] )
    # TODO: use bool instead.
    seq_data = binarize_categorical(seq_data, 4).astype(np.float32) # (num_nc, 4)
    # TODO: use other features
    # seq_data = np.concatenate([seq_data, other_feature], axis=1)
    seq_data = remove_nan(seq_data)
    nucleotide_attributes.append(seq_data)
  # TODO: how to use double chains? 
  #       combine to single chain or create 2 channel?
  nucleotide_attributes = np.concatenate(nucleotide_attributes)


  mapping = {}
  mapping.update(dict(
    exp_id = input_complex["exp_id"],
    protein = input_protein,
    nc_sequences = None,
    seqs_kmers = seqs_kmers,
    aa_attributes = aa_attributes,  # (num_aa, 20)
    aa_pwm = None,        # (num_aa, 20 or 30)
    aa_chm = aa_chm_feature,        # (num_aa, 7)
    atom_attributes = feature_by_atom, # (num_aa, MAX_NUM_ATOMS)
    aa_lengths = aa_lengths, # per aa length
    nucleotide_attributes = nucleotide_attributes, # (num_nc', 4) num_nc' = sum(all chain)
    nucleotide_other_attributes = None, # (num_nc', 10) 
    label         = label_dG,     # float scalar
    bin_ctrs      = None,     # np.arr (n_bins, )
    bin_half_w    = None,   # np.arr (n_bins, )
    label_bin     = None,    # int scalar
    label_offset  = None, # float scalar
  ))
  return mapping



def get_dd_instance(input_complex, args, other_inputs):
  input_protein    = input_complex["protein_sequence"] # string
  input_nucleotide = input_complex["nucleotide_sequence"] # string
  label_dG         = input_complex['dG'] if 'dG' in input_complex else input_complex['zscore']

  ################# get nucleotide features #################
  # nucleotide_to_index = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
  seqs_kmers = None
  nucleotide_sequences = None
  if '|' in input_nucleotide:
    nucleotide_sequences = input_nucleotide.split('|')
  else:
    nucleotide_sequences = [input_nucleotide]
  
  seqs_kmers = convert_to_conv_format(nucleotide_sequences, 
                                      kmers=args.kmers)

  mapping = {}
  mapping.update(dict(
    protein     = input_protein, # use by esm
    seqs_kmers  = seqs_kmers, # used by dbert
    label       = label_dG,     # float scalar
  ))
  return mapping


