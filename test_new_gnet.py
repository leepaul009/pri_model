import sys
import os
import collections
import pandas as pd
import numpy as np
import torch

from preprocessing.protein_chemistry import list_atoms,list_atoms_types,VanDerWaalsRadii,atom_mass,atom_type_to_index,atom_to_index,index_to_type,atom_type_mass
from preprocessing.protein_chemistry import residue_dictionary,hetresidue_field
from preprocessing import sequence_utils


from modeling.graph.frames import get_aa_frameCloud, get_atom_frameCloud

from modeling.graph.neighborhoods import FrameBuilder, LocalNeighborhood



def readPdbFile(file_path):
  r"""
  read a pdb file
  """
  if not os.exsits(file_path):
    print("error: following file not exists, {}".format(file_path))
    return
  #处理pdb文本，转为dataframe
  with open(file = file_path, mode ='r') as f1:
    data = f1.read()
    data = data.split('\n')
    del data[-3:]

  pdb = []
  for i in range(len(data)):
    element  = data[i].split()
    pdb.append(element)

  input = pd.DataFrame(pdb)
  #定义存放结果的字典
  amino_dict = collections.OrderedDict()
  atom_dict= collections.OrderedDict()

  for i in range(len(input)):
    #判断是否是H原子
    if input.loc[i,11] != 'H':
      atom_coord = np.array(input.loc[i,6:8].values,dtype= np.float64)
      atom_name = input.loc[i,2]
      atom_dict[atom_name] = atom_coord
    #判断是否为该pdb文件的最后一个原子
    if i == len(input)-1:
      amino_name = str(input.loc[i,5]) + '_' + input.loc[i, 3]
      amino_dict[amino_name] = atom_dict
      atom_dict= collections.OrderedDict()
    #非最后一个原子情况下判断是否为该氨基酸最后一个原子
    else:
      if input.loc[i,5] != input.loc[i+1,5]:
        amino_name = str(input.loc[i,5]) + '_' + input.loc[i, 3]
        amino_dict[amino_name] = atom_dict
        atom_dict= collections.OrderedDict()
  return amino_dict

def processDataPdbFormat(amino_dict):
  sequence = ""
  all_coordinates = []
  all_atoms = []
  all_atom_types = []
  for aa_key, atom_dict in amino_dict.items():
    _, aa_name = aa_key.split("_")
    sequence += residue_dictionary[aa_name]
    # List((3,)) ==> (atoms, 3)
    residue_atom_coordinates = np.stack([coord for _, coord in atom_dict.items()], axis=0)
    # (atoms,)
    residue_atoms = [atom_to_index[atom_name] for atom_name in atom_dict.keys()]
    residue_atom_type = [atom_type_to_index[atom_name[0]] for atom_name in atom_dict.keys()]

    all_coordinates.append(residue_atom_coordinates)
    all_atoms.append(residue_atoms)
    all_atom_types.append(residue_atom_type)

  return sequence, all_coordinates, all_atoms, all_atom_types

def getDataPdbFormat(file_paths):
  batch_sequences = []
  batch_all_coordinates = []
  batch_all_atoms = []
  batch_all_atom_types = []
  for file_path in file_paths:
    amino_dict = readPdbFile(file_path)
    sequence, all_coordinates, all_atoms, all_atom_types = processDataPdbFormat(amino_dict)

    batch_sequences.append(sequence)
    batch_all_coordinates.append(all_coordinates)
    batch_all_atoms.append(all_atoms)
    batch_all_atom_types.append(all_atom_types)

  return batch_sequences, batch_all_coordinates, batch_all_atoms, batch_all_atom_types

def binarize_categorical(matrix, n_classes, out=None):
  L = matrix.shape[0]
  matrix = matrix.astype(np.int32)
  if out is None:
    out = np.zeros([L, n_classes], dtype=np.bool_)
  subset = (matrix>=0) & (matrix<n_classes)
  out[np.arange(L)[subset],matrix[subset]] = 1
  return out


file_paths = ["dataset/P44_relaxed_rank_002_alphafold2_ptm_model_2_seed_000.pdb",]
batch_sequences, batch_all_coordinates, batch_all_atoms, batch_all_atom_types = getDataPdbFormat(file_paths)

sequence = batch_sequences[0]
all_coordinates, all_atoms = batch_all_coordinates[0], batch_all_atoms[0]

aa_clouds, aa_triplets, aa_indices = get_aa_frameCloud(all_coordinates, all_atoms)

aa_attributes = binarize_categorical(
    sequence_utils.seq2num(sequence)[0], 20)

atom_clouds, atom_triplets, atom_attributes, atom_indices = \
    get_atom_frameCloud(sequence, all_coordinates, all_atoms)



tensor_aa_clouds   = torch.Tensor(aa_clouds).unsqueeze(0)
tensor_aa_triplets = torch.Tensor(aa_triplets).unsqueeze(0)
tensor_aa_triplets = tensor_aa_triplets.long()
tensor_aa_indices  = torch.Tensor(aa_indices).unsqueeze(0)

config = None
frame_builder = FrameBuilder(config)

## put into data preprocess as multi-thread running
frames = frame_builder( [tensor_aa_clouds, tensor_aa_triplets] )



config = None
coordinates=['euclidian', 'index_distance', 'ZdotZ', 'ZdotDelta']
local_neighborhood = LocalNeighborhood(config, 
                                       Kmax=16, 
                                       coordinates=coordinates, 
                                       self_neighborhood=True, 
                                       index_distance_max=8, 
                                       nrotations=1)

tensor_aa_attributes = torch.Tensor(aa_attributes)
tensor_aa_attributes = tensor_aa_attributes.unsqueeze(0)

output = local_neighborhood( [frames, tensor_aa_indices, tensor_aa_attributes] )

neighbor_coordinates, neighbors_attributes = output[0][0], output[1]

neighbor_coordinates.shape, neighbors_attributes.shape

