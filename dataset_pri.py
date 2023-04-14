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

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

import utils
from preprocessing.protein_chemistry import list_aa, \
  aa_to_index, dictionary_covalent_bonds, list_atoms, atom_type_mass


def pri_get_instance(data_frame, file_name, args):
  mapping = {}
  mapping['file_name'] = file_name

  pkey = "protein_sequence"
  rkey = ""

  vectors = []

  # all_seq_by_str = [] # list of list of str
  # all_seq_by_int = [] # list of list of int

  for i in len(data_frame): # per chain
    prot_seq = data_frame.loc[i][pkey] # string
    rdna_seq = data_frame.loc[i][rkey] # string

    seq_by_int = [aa_to_index[it] # list_aa.index(it) 
      if it in list_aa else it.upper() # handle lower case of aa
      for it in prot_seq 
      if it in list_aa or it.upper() in list_aa] # check if aa not in list_aa

    # all_seq_by_int.append(seq_by_int)
    # all_seq_by_str.append([it_aa for it_aa in prot_seq])
    # atoms
    for it in seq_by_int: # per residue
      aa = list_aa[it]
      atoms = dictionary_covalent_bonds[aa] # dict of atoms
      # we could get atom's feature from outside
      aa_atoms = list(atoms.keys()) # list of string

      for atom in aa_atoms:
        atom_type_int = list_atoms.index(atom) # index in the atom table
        mass = atom_type_mass[atom_type_int]
        vector = [atom_type_int, mass]


  # mapping.update(dict(
  #   matrix=matrix, # feat [n_nodes(traj + map), 128]
  #   labels=np.array(labels).reshape([30, 2]), # gt traj
  #   polyline_spans=[slice(each[0], each[1]) for each in polyline_spans], # list of slice(slice记录一个polyline的再matrix中的起止位置)
  #   labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64), # gt validance [30, 2]
  #   eval_time=30,
  # ))

  return mapping


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
      data_dir = args.data_dir
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

        pbar = tqdm(total=len(files))

        queue = multiprocessing.Queue(args.core_num) # inputs container
        queue_res = multiprocessing.Queue() # outputs container

        def calc_ex_list(queue, queue_res, args):
          res = []
          dis_list = []
          while True:
              file = queue.get()
              if file is None:
                  break
              if file.endswith("csv"):
                  # with open(file, "r", encoding='utf-8') as fin:
                  #     lines = fin.readlines()[1:]
                  data_frame = pd.read_csv(file, sep='\t')
                  # read and process each file
                  instance = pri_get_instance(data_frame, file, args)
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
        for file in files:
            assert file is not None
            queue.put(file)
            pbar.update(1)

        # necessary because queue is out-of-order
        while not queue.empty():
            pass
        pbar.close()
        
        self.ex_list = []

        pbar = tqdm(total=len(files))
        for i in range(len(files)):
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