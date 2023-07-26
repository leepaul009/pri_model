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
  input_protein = input_complex["protein_sequence"] # string
  input_nucleotide = input_complex["nucleotide_sequence"] # string
  # label_dG = input_complex['zscore']
  label_dG = input_complex['dG'] if 'dG' in input_complex else input_complex['zscore']

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


  mapping = {}
  mapping.update(dict(
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



class PriDataset(torch.utils.data.Dataset):
  def __init__(self, args, data_dir, batch_size, to_screen=True):
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
      print("Create dataset to following files, len = {}, first = {}".format(len(files), files[0]))


      # data_frame = pd.read_csv(files[0], sep='\t')
      # num_lines = len(data_frame)
      # pbar = tqdm(total=num_lines)
      # pbar = tqdm(total=len(files))

      data_frame_list = []
      num_lines = 0
      for f in files:
        df = pd.read_csv(f, sep='\t')
        if args.debug:
          debug_len = min(120, len(df))
          df = df.loc[:debug_len]
          print("debug mode: only use {} items".format(debug_len))
        num_lines += len(df)
        data_frame_list.append(df)
      pbar = tqdm(total=num_lines)

      # other inputs:
      # other_data_dict = {}
      # if args.use_chemistry:
      #   other_data_f = os.path.join('data', 'other_feat.npy')
      #   if (os.path.exists(other_data_f)):
      #     other_data_dict = np.load(other_data_f, allow_pickle=True).item()

      queue = multiprocessing.Queue(args.core_num) # inputs container
      queue_res = multiprocessing.Queue() # outputs container

      def calc_ex_list(queue, queue_res, args):
        res = []
        dis_list = []
        while True:
          # get one item(one row) from queue
          other_inputs = dict()
          # if args.use_chemistry:
          #   data_item, other_inputs = queue.get()
          # else:
          data_item = queue.get()
          if data_item is None:
              break
          # with open(file, "r", encoding='utf-8') as fin:
          #     lines = fin.readlines()[1:]
          # instance: dict
          if args.data_name == 'hox_data':
            instance = hox_get_instance(data_item, args, other_inputs)
          else:
            instance = pri_get_instance(data_item, args, other_inputs)
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
      # for i in range(num_lines):
      #   assert data_frame.loc[i] is not None
      #   # insert one row into queue
      #   queue.put(data_frame.loc[i])
      #   pbar.update(1)
      
      for df in data_frame_list:
        for i in range(len(df)):
          assert df.loc[i] is not None
          # if args.use_chemistry:
          #   other_inputs = {
          #     'other_feat' : other_data_dict[ df.loc[i]['complex_id'] ] # np.arr
          #   }
          #   queue.put( (df.loc[i], other_inputs) )
          # else:
          queue.put(df.loc[i])
          pbar.update(1)

      # necessary because queue is out-of-order
      while not queue.empty():
          pass
      pbar.close()
      
      # move output files from queue_res into self.ex_list
      self.ex_list = []
      pbar = tqdm(total=num_lines)
      for i in range(num_lines):
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


class PriDatasetExt(torch.utils.data.Dataset):
  def __init__(self, args, data_dir, batch_size, to_screen=True):
    if not isinstance(data_dir, list):
      data_dir = [data_dir]
    self.ex_list = []
    self.args = args
    self.batch_size = batch_size

    files = []
    for each_dir in data_dir:
      root, dirs, cur_files = os.walk(each_dir).__next__()
      files.extend([os.path.join(each_dir, file) for file in cur_files 
                    if file.endswith("csv") and not file.startswith('.')])
    print("Create dataset to following files: len = {}".format(len(files)))

    data_frame_list = []
    num_lines = 0
    for f in files:
      df = pd.read_csv(f, sep='\t')
      if args.debug:
        debug_len = min(640, len(df))
        df = df.loc[:debug_len]
        print("debug mode: only use {} items".format(debug_len))
      num_lines += len(df)
      data_frame_list.append(df)
    # pbar = tqdm(total=num_lines)

    self.merged_df = None
    if len(data_frame_list) > 1:
      self.merged_df = pd.concat(data_frame_list).reset_index()
    else:
      self.merged_df = data_frame_list[0]
    print("Read all dataframe with length = {}".format(len(self.merged_df)))

    prot_chm_f = 'dataset/prot_chm_data2.txt'
    prot_chm_df = pd.read_table(prot_chm_f, index_col=0)
    self.other_inputs = dict()
    # df.loc['G','Steric_parameter']
    self.other_inputs['prot_chm_df'] = prot_chm_df


  def __len__(self):
    # return len(self.ex_list)
    return len(self.merged_df)

  def __getitem__(self, idx):
    instance = self.merged_df.loc[idx]

    if self.args.data_name == 'hox_data':
      output = hox_get_instance(instance,  self.args, self.other_inputs)
    else:
      output = pri_get_instance(instance, self.args, self.other_inputs)
    
    return output



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
from utilities.comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()


  torch.cuda.set_device(args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  train_dataset = PriDataset(args, args.data_dir, args.train_batch_size, to_screen=False)
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



