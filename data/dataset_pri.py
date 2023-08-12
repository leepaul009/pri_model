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

from data.dataset_instance import pri_get_instance, hox_get_instance



class PriDatasetOld(torch.utils.data.Dataset):
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

## optimized
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
        root, dirs, cur_files = os.walk(each_dir).__next__()
        files.extend([os.path.join(each_dir, file) for file in cur_files 
                      if file.endswith("csv") and not file.startswith('.')])
      print("Create dataset to following files, len = {}, first = {}".format(len(files), files[0]))

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

      queue = multiprocessing.Queue(args.core_num) # inputs container
      queue_res = multiprocessing.Queue() # outputs container

      def calc_ex_list(queue, queue_res, args):
        res = []
        dis_list = []
        while True:
          other_inputs = dict()
          data_item = queue.get() # get and remove from queue
          if data_item is None:
              break
          if args.data_name == 'hox_data':
            instance = hox_get_instance(data_item, args, other_inputs)
          else:
            instance = pri_get_instance(data_item, args, other_inputs)
          if instance is not None:
              # data_compress = zlib.compress(pickle.dumps(instance))
              # res.append(data_compress)
              # queue_res.put(data_compress)
              outfdir = os.path.join('data/tmp/')
              if not os.path.exists(outfdir):
                os.makedirs(outfdir)
              outf = os.path.join(outfdir, '{}.pkl'.format(instance['exp_id']))
              if not os.path.exists(outf) or not args.direct_read_cache:
                with open(outf, 'wb') as f:
                  pickle.dump(instance, f)
              queue_res.put( (outf, instance['label']) )
          else:
              queue_res.put(None)
      
      processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) 
                    for _ in range(args.core_num)]
      for each in processes:
          each.start()
      
      for df in data_frame_list:
        for i in range(len(df)):
          assert df.loc[i] is not None
          queue.put(df.loc[i])
          pbar.update(1)

      # necessary because queue is out-of-order
      # wait items in queue be used and removed
      while not queue.empty():
        pass
      pbar.close()
      
      # move output files from queue_res into self.ex_list
      self.ex_list = []
      self.labels = []
      pbar = tqdm(total=num_lines)
      for i in range(num_lines):
        res = queue_res.get()
        if res is not None:
          t, l = res
          self.ex_list.append(t)
          self.labels.append(l)
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
      file = self.ex_list[idx]
      pickle_file = open(file, 'rb')
      instance = pickle.load(pickle_file)
      pickle_file.close()

      # data_compress = self.ex_list[idx]
      # instance = pickle.loads(zlib.decompress(data_compress))
      return instance



class PriDatasetExt(torch.utils.data.Dataset):
  def __init__(self, args, data_dir, batch_size, to_screen=True):
    if not isinstance(data_dir, list):
      data_dir = [data_dir]
    self.ex_list = []
    self.labels = []
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

    if 'dG' in self.merged_df.columns:
      self.labels = self.merged_df['dG'].values
    else:
      self.labels = self.merged_df['zscore'].values

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



