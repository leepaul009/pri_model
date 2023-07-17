import numpy as np
import pandas as pd
import argparse

MAX_PROTEIN_LENGTH = 1000
MAX_NUCLEOTIDE_LENGTH = 150




def split_dataset(data, targets, valid_frac=0.2):

  ind0 = np.where(targets<0.5)[0]
  ind1 = np.where(targets>=0.5)[0]
  
  n_neg = int(len(ind0)*valid_frac)
  n_pos = int(len(ind1)*valid_frac)

  shuf_neg = np.random.permutation(len(ind0))
  shuf_pos = np.random.permutation(len(ind1))

  X_train = np.concatenate((data[ind1[shuf_pos[n_pos:]]], data[ind0[shuf_neg[n_neg:]]]))
  Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))
  train = (X_train, Y_train)

  X_test = np.concatenate((data[ind1[shuf_pos[:n_pos]]], data[ind0[shuf_neg[:n_neg]]]))
  Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))
  test = (X_test, Y_test)

  return train, test


def split_dataframe(data_frame : pd.DataFrame, valid_frac=0.2):
  length = len(data_frame)
  ind = np.arange(length)
  shuf_ind = np.random.permutation(length)
  
  n_train = int(length*0.5)
  n_val = int(length*0.25)
  
  data_train = data_frame.loc[ind[shuf_ind[:n_train]]]
  data_val   = data_frame.loc[ind[shuf_ind[n_train:n_train+n_val]]]
  data_test  = data_frame.loc[ind[shuf_ind[n_train+n_val:]]]

  return data_train, data_val, data_test


def GenerateData(file):
  df = pd.read_csv(file, sep='\t')
  input_size = len(df)

  # same protein sequence corresponds to multiple inputs
  map_protein2size = {} # certain protein has how many data
  map_protein2length = {} # certain protein's length

  map_nucleotide2size = {}
  map_nucleotide2length = {}

  num_unique_protein = 0
  num_unique_nucleotide = 0

  for i in range(len(df)):
    seq = df.loc[i]['protein_sequence']
    if seq not in map_protein2size:
      map_protein2size[seq] = 1
      map_protein2length[seq] = len(seq)
      num_unique_protein += 1
    else:
      map_protein2size[seq] += 1
    
    seq = df.loc[i]['nucleotide_sequence']
    # if "|" in seq:
    #   seq = seq.replace("|", "")
    if seq not in map_nucleotide2size:
      map_nucleotide2size[seq] = 1
      map_nucleotide2length[seq] = len(seq)
      num_unique_nucleotide += 1
    else:
      map_nucleotide2size[seq] += 1

  print("there are {} unique proteins, {} unique DNA/RNA".format(
    num_unique_protein, num_unique_nucleotide))


  # protein whose length is larger than MAX_PROTEIN_LENGTH will be used as val set and test set
  prot_keys = list(map_protein2length.keys())
  prot_keys = np.array(prot_keys) # (num_unique_seq, )
  prot_vals = list(map_protein2length.values())
  prot_vals = np.array(prot_vals) # (num_unique_seq, )
  size_by_keys = np.array([map_protein2size[k] for k in prot_keys]) # (num_unique_seq, )

  indices = np.where(prot_vals > MAX_PROTEIN_LENGTH)[0]
  long_prot_keys = prot_keys[indices]
  print("there are {} items of data, whose protein length > {}".format(
    np.sum(size_by_keys[indices]), MAX_PROTEIN_LENGTH))


  # DNA/RNA whose length is larger than MAX_PROTEIN_LENGTH will be used as val set and test set
  nc_keys = list(map_nucleotide2length.keys())
  nc_keys = np.array(nc_keys) # (num_unique_seq, )
  nc_vals = list(map_nucleotide2length.values())
  nc_vals = np.array(nc_vals) # (num_unique_seq, )
  size_by_keys_nc = np.array([map_nucleotide2size[k] for k in nc_keys]) # (num_unique_seq, )

  indices = np.where(nc_vals > MAX_NUCLEOTIDE_LENGTH)[0]
  long_nc_keys = nc_keys[indices]
  print("there are {} items of data, whose DNA/RNA length > {}".format(
    np.sum(size_by_keys_nc[indices]), MAX_NUCLEOTIDE_LENGTH))


  orig_labels = np.array(df.dG)

  indices = []
  indices_long = []
  for i in range(len(df)):
    aa_seq = df.loc[i]['protein_sequence']
    nc_seq = df.loc[i]['nucleotide_sequence']
    if aa_seq not in long_prot_keys and nc_seq not in long_nc_keys:
      indices.append(i)
    else:
      indices_long.append(i)
  indices = np.array(indices)
  indices_long = np.array(indices_long)

  df_test_set = df.loc[indices_long] # return a new pd.DataFrame

  new_df = df.loc[indices] # return a new pd.DataFrame
  new_df.reset_index(inplace = True, drop = True)
  labels = orig_labels[indices]

  
  intervals = np.arange(0, -30, -2).astype(np.float32)
  inds_groups = []
  for i in range(intervals.shape[0]-1):
    temp = np.logical_and(labels > intervals[i], labels <= intervals[i-1])
    inds = np.where(temp)[0]
    if inds.shape[0] > 0:
      inds_groups.append(inds)
    # print("dG < {} has size = {}".format(intervals[i], len(indices)))

  data_trains = []
  data_vals = []
  data_tests = []
  for inds in inds_groups:
    temp_df = new_df.loc[inds]
    temp_df.reset_index(inplace = True, drop = True)
    data_train, data_val, data_test = split_dataframe(temp_df)
    print("this group have {} train_data, {} val_data, {} test_data".format(
      len(data_train), len(data_val), len(data_test)))
    data_trains.append(data_train)
    data_vals.append(data_val)
    data_tests.append(data_test)

  data_train = pd.concat(data_trains)
  data_val = pd.concat(data_vals)
  data_test = pd.concat(data_tests)
  data_test = pd.concat([data_test, df_test_set])

  print("finally we have {} train_data, {} val_data, {} test_data".format(
    len(data_train), len(data_val), len(data_test)))

  data_train.to_csv('data/debug/train.csv', index=False, sep='\t')
  data_val.to_csv('data/debug/val.csv', index=False, sep='\t')
  data_test.to_csv('data/debug/test.csv', index=False, sep='\t')

  # data_train.to_csv('data/train/train.csv', index=False, sep='\t')
  # data_val.to_csv('data/val/val.csv', index=False, sep='\t')
  # data_test.to_csv('data/test/test.csv', index=False, sep='\t')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", default="data/_datasets/seq_dg_cluster_final_230607_v2.csv", type=str)
  args = parser.parse_args()
  # data_file = 'data/_datasets/seq_dg_identity_230404.csv'

  GenerateData(args.data)







