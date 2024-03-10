import os
from typing import Sequence, Tuple, List, Union, Dict
import pickle
import zlib

import torch

from data.graph.dataset import \
  readPdbFile, processDataPdbFormat, binarize_categorical
from preprocessing import sequence_utils

from modeling.graph.frames import get_aa_frameCloud, get_atom_frameCloud
from modeling.graph.neighborhoods import FrameBuilder


class GraphBatchConvert(object):
  def __init__(self, seq2file=None, default_pdb_file=None):
    self.default_pdb_file = default_pdb_file
    # self.pdb_files = [default_pdb_file] if pdb_files is None else pdb_files
    self.seq2file = seq2file
    self.frame_builder = FrameBuilder(config=None)

  def process(self, file_path):
    amino_dict = readPdbFile(file_path)
    # sequence: string
    # all_coordinates: list[numpy.array], 
    #   list indicates amino acids
    #   each numpy.array have a shape of (n,3) where n is 
    #   number of atoms in corresponding amino acid
    # all_atoms: List[List[int]]
    #   each element is atom index
    #   where outer list indicates amino acids
    #   and inner list indicates atoms
    # all_atom_types: List[List[int]]
    #   each element is index of atom type
    #   where outer list indicates amino acids
    #   and inner list indicates atoms 
    sequence, all_coordinates, all_atoms, all_atom_types \
      = processDataPdbFormat(amino_dict)

    # protein_feat = binarize_categorical(
    #   sequence_utils.seq2num(sequence)[0], 20) # (num_aa, 20)

    # size of num_aa×2+1
    # aa_clouds shape=(num_aa×2+1, 3)
    # aa_triplets shape=(num_aa, 3)
    # all_atoms shape=(num_aa, 1)
    aa_clouds, aa_triplets, aa_indices = \
      get_aa_frameCloud(all_coordinates, all_atoms)

    # not used now
    atom_clouds, atom_triplets, atom_attributes, atom_indices = \
      get_atom_frameCloud(sequence, all_coordinates, all_atoms)

    tensor_aa_clouds   = torch.Tensor(aa_clouds).unsqueeze(0)
    tensor_aa_triplets = torch.Tensor(aa_triplets).unsqueeze(0).long()

    # (1, num_aa, 4, 3)
    frame = self.frame_builder(
      [tensor_aa_clouds, tensor_aa_triplets])
    frame = frame.squeeze(0) # (num_aa, 4, 3)
    # tensor_protein_feat = torch.Tensor(protein_feat).unsqueeze(0) # (1, num_aa, 20)

    aa_indices = torch.Tensor(aa_indices)

    # frame: (num_aa, 4, 3)
    return frame, aa_indices

  def __call__(self, inputs: List[Dict]):
    
    list_protein_feats = []
    list_frames = []
    list_aa_indices = []
    labels = []

    max_length = 0
    batch_size = 0

    for it in inputs:
      protein = pickle.loads(zlib.decompress( it["protein"] )) 
      dna     = pickle.loads(zlib.decompress( it["dna"] )) 
      label   = pickle.loads(zlib.decompress( it["label"] ))

      # TODO: check protein is string
      assert isinstance(protein, str), "{} is not string".format(protein)
      assert isinstance(dna, str), "{} is not string".format(dna)
      assert isinstance(label, float), "{} is not float".format(label)

      length_protein = len(protein)
      max_length = max([length_protein, max_length])

      if self.seq2file is not None:
        file_path = self.seq2file[protein]
      else:
        file_path = self.default_pdb_file

      frame, aa_indices = self.process(file_path)
      # 现在的 frame 和 aa_indices 是假的，所以序列长度也是假的，需要这里改长度
      frame = frame[:length_protein] # (num_aa, 4, 3)
      aa_indices = aa_indices[:length_protein] # (num_aa, 1)

      batch_size += 1

      protein_feat = binarize_categorical(
        sequence_utils.seq2num(protein)[0], 20) # (num_aa, 20)
      protein_feat = torch.Tensor(protein_feat).float()

      assert frame.shape[0] == protein_feat.shape[0]

      list_protein_feats.append(protein_feat)
      list_frames.append(frame)
      list_aa_indices.append(aa_indices)
      labels.append(label)

    # 这里的tensor 是经过 padding 过的
    tensor_frames = torch.zeros(batch_size, max_length, 4, 3).float()
    tensor_feats  = torch.zeros(batch_size, max_length, 20).float()
    tensor_aa_indices = torch.zeros(batch_size, max_length, 1).float()

    for i in range(batch_size):
      frame, indices, feat = list_frames[i], list_aa_indices[i], list_protein_feats[i]
      L = feat.shape[0]
      tensor_frames[i, :L] = frame
      tensor_feats[i, :L]  = feat
      tensor_aa_indices[i, :L] = indices
    
    labels = torch.Tensor(labels).long()

    # feat:    (bs, seq(pad), 20)
    # frame:   (bs, seq(pad), 4, 3)
    # indices: (bs, seq(pad), 1)
    # label:   (bs)
    return tensor_feats, tensor_frames, tensor_aa_indices, labels


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