import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import argparse
import multiprocessing
from multiprocessing import Process
import pickle
import random
import zlib
import copy



# set env
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 100)

### why same protein have two or more probe_intensity_file

G_NUM_TEST = 0.2
G_NUM_VAL = 0.2
G_NUM_TRAIN = 0.6

def createDir(p):
  if not os.path.exists(p):
    print("Directory {} doesn't exist, create a new.".format(p))
    os.makedirs(p)


def get_instance(subdf):
  assert subdf is not None, 'input is not valid'
  root_dir = 'dataset/dna_dataset'
  sub_dir = 'relative_dna_dnaseq_intensity'

  pseq = subdf['protein_sequence'].values[0]
  motId = subdf['Motif_ID'].values[0]
  assert len(np.unique(subdf['protein_sequence'].values)) == 1
  ### get dna seqs
  dna_seqs, vals = list(), list() 
  for f in subdf['probe_intensity_file'].values:
    subf = os.path.join(root_dir, sub_dir, f)
    f = open(subf, 'r')
    lines = f.readlines()
    # print(len(lines))
    for line in lines:
      # print(line.rstrip().split('\t'))
      tmp = line.rstrip().split('\t')
      v = float(tmp[1])
      if not isinstance(v, float):
        v = 0.0
      dna_seqs.append(str(tmp[0]))
      vals.append(v)
  ###
  dna_seqs  = np.array(dna_seqs)
  vals      = np.array(vals)
  # all_vals.append(vals)
  sz = len(vals)
  pseqs = np.array([pseq])
  pseqs = np.repeat(pseqs, sz)

  tmpdf = pd.DataFrame({
      'protein_sequence': pseqs,
      'nucleotide_sequence': dna_seqs,
      'dG': vals,
  })
  # lst_df_by_pseq.append(tmpdf)
  return tmpdf, motId

def process_dna_dataset(args, current_path):
  root_dir = os.path.join(current_path, 'dataset/dna_dataset')
  sub_dir = 'relative_dna_dnaseq_intensity'
  madf = pd.read_csv(os.path.join(root_dir, 'relative_dna_proteinseq.txt'), sep='\t')
  
  unique_pseq = np.unique(madf['protein_sequence'].values)
  num_dfs = len(unique_pseq)
  

  list_data_by_prot = []

  if args.core_num > 1:
    queue = multiprocessing.Queue(args.core_num) # inputs container
    queue_res = multiprocessing.Queue()
    pbar = tqdm(total=num_dfs)
    
    def calc_list(queue, queue_res, args):
      res = []
      while True:
        data_item = queue.get()
        if data_item is None:
          break
        instance = get_instance(data_item)
        if instance is not None:
          data_compress = zlib.compress(
            pickle.dumps(instance))
          res.append(data_compress)
          queue_res.put(data_compress)
        else:
          queue_res.put(None)
    
    processes = [Process(target=calc_list, 
                        args=(queue, queue_res, args)) 
                  for _ in range(args.core_num)]
    for each in processes:
      each.start()

    # map_df_by_pseq = dict()
    sum_id = 0
    list_subdf = list()
    for k, subdf in madf.groupby('protein_sequence'):
      queue.put(subdf)
      pbar.update(1)
      sum_id += 1
    assert sum_id == num_dfs, 'invalid number of unique prot'

    while not queue.empty():
      pass
    pbar.close()

    
    pbar = tqdm(total=num_dfs)
    for i in range(num_dfs):
      t = queue_res.get()
      if t is not None:
          list_data_by_prot.append(t)
      pbar.update(1)
    pbar.close()
    pass

    for i in range(args.core_num):
        queue.put(None)
    for each in processes:
        each.join()
  


  ### done
  all_vals = list()
  for data_compress in list_data_by_prot:
    df, _ = zlib.decompress(data_compress)
    all_vals.append(df['dG'].values)  
  all_vals = np.concatenate(all_vals).reshape(-1)

  vmean, vstd = all_vals.mean(), all_vals.std()
  # vmean = 4834.18
  # vstd = 112976855.02

  # G_NUM_TEST = 0.2
  # G_NUM_VAL = 0.2
  # G_NUM_TRAIN = 0.6

  train_dir = os.path.join(root_dir, '_dataset', 'train')
  createDir(train_dir)
  val_dir = os.path.join(root_dir, '_dataset', 'val')
  createDir(val_dir)
  test_dir = os.path.join(root_dir, '_dataset', 'test')
  createDir(test_dir)

  print("begin to write files")
  for i, data_compress in tqdm(enumerate(list_data_by_prot)):
    df, motId = zlib.decompress(data_compress)
    df.loc[:, 'dG'] = df['dG'].apply(
      lambda x : (x - vmean ) / (vstd + 1e-5) )
    
    sz = len(df)
    inds = np.arange(sz)
    shuf_inds = np.random.permutation(sz)
    test_sz =  int(np.ceil(sz * G_NUM_TEST))
    train_sz = int(np.ceil(sz * G_NUM_TRAIN))
    # val_sz = sz - test_sz - train_sz

    test_df  = df.loc[inds[shuf_inds[:test_sz]]]
    train_df = df.loc[inds[shuf_inds[test_sz:(test_sz + train_sz)]]]
    val_df   = df.loc[inds[shuf_inds[(test_sz + train_sz):]]]

    f = os.path.join(test_dir, 'test_{}.csv'.format(motId))
    test_df.to_csv(f, index=False, sep='\t')

    f = os.path.join(train_dir, 'train_{}.csv'.format(motId))
    train_df.to_csv(f, index=False, sep='\t')

    f = os.path.join(val_dir, 'val_{}.csv'.format(motId))
    val_df.to_csv(f, index=False, sep='\t')

  return True



if __name__ == '__main__':
  ###
  parser = argparse.ArgumentParser()
  parser.add_argument('--core_num', type=int, default=8)

  args = parser.parse_args()
  current_path = os.getcwd()
  print("当前路径为：%s" % current_path)

  process_dna_dataset(args, current_path)