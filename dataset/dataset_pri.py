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
  nucleotide_to_index


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

# input:
#   input_complex: one row of data
#   args:
# output:
#   a dict containing data of one prot complex
def pri_get_instance(input_complex, args):
  """
  Process one input.

  Arguments:
  - input_complex: one complex containing a amino acid sequence and nucleotide sequence
  
  Returns:
  map of input features and label.
  """
  mapping = {}

  input_protein = input_complex["protein_sequence"] # string
  input_nucleotide = input_complex["nucleotide_sequence"] # string
  label_dG   = input_complex['dG']


  protein_sequences = process_sequence_without_coordinates(input_protein)
  num_aa = len(protein_sequences)

  # TODO: use bool instead.
  aa_attributes = binarize_categorical(
    protein_sequences, 20).astype(np.float32) # (seq_aa, 20)
  aa_indices = np.array([i for i in range(num_aa)])

  atom_attributes = []
  atom_mass_attributes = []
  atom_indices = []

  for i, it in enumerate(protein_sequences): # per aa
    aa = list_aa[it] # get aa string
    aa_atoms = list(dictionary_covalent_bonds[aa].keys())
    num_atoms = len(aa_atoms)
    atom_type = np.array([list_atoms.index(x) for x in aa_atoms])
    atom_mass = np.array([atom_type_mass[x] for x in atom_type])
    
    atom_type = atom_type.reshape(-1,1) # (num_atoms, 1)
    atom_mass = atom_mass.reshape(-1,1) # (num_atoms, 1)
    
    atom_type = remove_nan(atom_type)
    atom_mass = remove_nan(atom_mass)
    
    atom_attributes.append(atom_type)
    atom_mass_attributes.append(atom_mass)

    # indicates each atom belong to the index of aa
    atom_indices.append(np.ones((num_atoms,), dtype=np.int32) * i)

  atom_indices = np.concatenate(atom_indices, axis=0)
  atom_indices = remove_nan(atom_indices)
  aa_attributes = remove_nan(aa_attributes)
  

  # process nucleotide data. nucleotide_to_index = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
  nucleotide_sequences = None
  if '|' in input_nucleotide:
    nucleotide_sequences = input_nucleotide.split('|')
  else:
    nucleotide_sequences = [input_nucleotide]
  
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

  mapping.update(dict(
    aa_attributes = aa_attributes, # (num_aa, 20)
    aa_indices = aa_indices, # (num_aa,)
    atom_attributes = atom_attributes, # list[np.ndarray=(num_atoms,1)]
    atom_indices = atom_indices, # (all_atoms_in_aa, )
    nucleotide_attributes = nucleotide_attributes, # (num_nc', 4) num_nc' = sum(all chain)
    label = label_dG, # float
  ))
  return mapping



class PriDataset(torch.utils.data.Dataset):
  def __init__(self, args, batch_size, to_screen=True):
    data_dir = args.data_dir
    if not isinstance(data_dir, list):
      data_dir = [data_dir]
    self.ex_list = []
    self.args = args
    self.batch_size = batch_size

    if args.core_num >= 1:
      files = []
      for each_dir in data_dir:
        # root is current dir, dirs is sub-dir of current dir, cur_files is files under current dir
        root, dirs, cur_files = os.walk(each_dir).__next__()
        # 
        files.extend([os.path.join(each_dir, file) for file in cur_files 
                      if file.endswith("csv") and not file.startswith('.')])
      print(files[:3])


      data_frame = pd.read_csv(files[0], sep='\t')
      num_lines = len(data_frame)
      num_lines = 64 * 10
      pbar = tqdm(total=num_lines)
      # pbar = tqdm(total=len(files))

      queue = multiprocessing.Queue(args.core_num) # inputs container
      queue_res = multiprocessing.Queue() # outputs container

      def calc_ex_list(queue, queue_res, args):
        res = []
        dis_list = []
        while True:
          # get one item(one row) from queue
          data_item = queue.get()
          if data_item is None:
              break
          # with open(file, "r", encoding='utf-8') as fin:
          #     lines = fin.readlines()[1:]
          # instance: dict
          instance = pri_get_instance(data_item, args)
          if instance is not None:
              data_compress = zlib.compress(pickle.dumps(instance))
              res.append(data_compress)
              queue_res.put(data_compress)
          else:
              queue_res.put(None)
      
      processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) 
                    for _ in range(args.core_num)]
      for each in processes:
          each.start()
      # res = pool.map_async(calc_ex_list, [queue for i in range(args.core_num)])
      for i in range(num_lines):
        assert data_frame.loc[i] is not None
        # insert one row into queue
        queue.put(data_frame.loc[i])
        pbar.update(1)

      # necessary because queue is out-of-order
      while not queue.empty():
          pass
      pbar.close()
      
      self.ex_list = []

      pbar = tqdm(total=num_lines)
      for i in range(num_lines):
        #
        t = queue_res.get()
        if t is not None:
            self.ex_list.append(t)
        pbar.update(1)
      pbar.close()
      pass

      for i in range(args.core_num):
          queue.put(None)
      for each in processes:
          each.join()
    else:
      assert False



  def __len__(self):
      return len(self.ex_list)

  def __getitem__(self, idx):
      # file = self.ex_list[idx]
      # pickle_file = open(file, 'rb')
      # instance = pickle.load(pickle_file)
      # pickle_file.close()

      data_compress = self.ex_list[idx]
      instance = pickle.loads(zlib.decompress(data_compress))
      return instance

# to be used
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def from_numpy(data):
  """Recursively transform numpy.ndarray to torch.Tensor.
  """
  if isinstance(data, dict):
      for key in data.keys():
          data[key] = from_numpy(data[key])
  if isinstance(data, list) or isinstance(data, tuple):
      data = [from_numpy(x) for x in data]
  if isinstance(data, np.ndarray):
      """Pytorch now has bool type."""
      data = torch.from_numpy(data)
  return data

# to be used
def worker_init_fn(pid):
    # np_seed = hvd.rank() * 1024 + int(pid)
    np_seed = get_rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


from torch.utils.data.distributed import DistributedSampler
from comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()


  torch.cuda.set_device(args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  train_dataset = PriDataset(args, args.train_batch_size, to_screen=False)
  train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
  # batch_list_to_batch_tensors collect one batch of inputs as List[Dict]
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler,
    batch_size=args.train_batch_size // get_world_size(),
    collate_fn=utils.batch_list_to_batch_tensors)

  for step, batch in enumerate(train_dataloader):
    print("step {}, batch.type={}".format( step, type(batch) ))
    if (isinstance(batch, list)):
      # batch has 64 inputs(list of dict)
      print("batch size={}".format( len(batch) )) 



