import copy
import math
import multiprocessing
import os
import pickle
import random
import zlib
import pandas as pd
from collections import defaultdict
from multiprocessing import Process
from random import choice
import argparse

import numpy as np
import torch
from tqdm import tqdm
list_aa = [
    'A',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'K',
    'L',
    'M',
    'N',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'V',
    'W',
    'Y'

]



  
def toBinLabel(dg_label, bin_ctrs, bin_half_w):
  found = False
  for i in range(len(bin_ctrs)):
    ctr = bin_ctrs[i]
    half_w = bin_half_w[i]
    
    bmin = ctr - half_w
    bmax = ctr + half_w
    if dg_label > bmin and dg_label <= bmax:
      offset = (dg_label - ctr) / half_w
      found = True
      return i, offset
  
  if not found:
    print("can not find bin label from bin info")
    
    
def processBinLabel(args, label_dG):
  '''
  -25 ~ -18
  -18 ~ -16
  -16 ~ -4 距离12, 宽度为2的则有6个类
  -16 -15 -14, -14 -13 -12
  10-11, 11-12, 12-13, 13-24
  0-3, 3-5, 5-7, 7-8, 8-9
  '''
  bin_ctrs, bin_half_w, label_bin, label_offset = None, None, None, None

  # bins = []
  # bins.append([-24., 4])
  # bins.append([-20., 4])
  # bins.append([-17., 2])
  # # (-16.5, 1) -> (-4.5, 1)
  # for x in np.arange(-16.5, -3.5, 1):
  #   bins.append([x, 1])
  # bins.append([-3., 2])
  # bins.append([-1., 2])
  # num_bins = len(bins)
  
  bin_info = {
    -1.3 : (-0.0, -3.6, 37),
    -5.7 : (-3.6, -6.6, 357),
    -7.5 : (-6.6, -8.3, 942),
    -9.0 : (-8.3, -9.6, 1178),
    -10.2 : (-9.6, -10.7, 1150),
    -11.3 : (-10.7, -12.1, 973),
    -12.8 : (-12.1, -13.8, 431),
    -14.8 : (-13.8, -25.0, 109),
  }
  
  list_ctr = []
  list_half_w = []
  for k, v in bin_info.items():
    mean_x = k
    max_x, min_x, len_x = v
    ctr = (max_x + min_x) / 2.0
    half_w = np.abs(max_x - min_x) / 2.0
    list_ctr.append(ctr)
    list_half_w.append(half_w)
  
  bin_ctrs = np.array(list_ctr) # (n_bins, )
  bin_half_w = np.array(list_half_w) # (n_bins, )
  
  sorted_ids = np.argsort(bin_ctrs)
  bin_ctrs = bin_ctrs[sorted_ids]
  bin_half_w = bin_half_w[sorted_ids]
  
  label_bin, label_offset = toBinLabel(label_dG, bin_ctrs, bin_half_w) # int, float

  return bin_ctrs, bin_half_w, label_bin, label_offset
  
def computeBin(df):
  bin_info = dict()
  # TODO: 
  dgs = df['dG'].values
  
  X = dgs.reshape(-1,1)
  # TODO: make n_cluster a parameters
  model = KMeans(n_clusters=8)
  
  model.fit(X)
  yhat = model.predict(X)
  clusters = np.unique(yhat)
  for cluster in clusters:
    row_ix = np.where(yhat == cluster)
    cur_x = X[row_ix]
    # cur_x.mean(), cur_x.max(), cur_x.min(), len(cur_x)
    bin_info[cur_x.mean()] = ( cur_x.max(), cur_x.min(), len(cur_x) )
    
  return bin_info
  
  
def getProtChemFeature(dir, input_complex, args):
  prot_key = input_complex['protein_index']
  f = "{}.txt".format(prot_key)
  f = os.path.join(dir, f)
  df = pd.read_csv(f, sep='\t')
  
  cols = ["Steric_parameter", "polarizability",	"volume",	
          "hydrophobicity",	"isoelectric point",	
          "Helix probability",	"Sheet probability"]
  chem_feat = df[cols].values # (num_aa, 7)
  
  return chem_feat

  # cols = ["Steric_parameter", "polarizability",	"volume",	"hydrophobicity",	"isoelectric point",	
  #              "Helix probability",	"Sheet probability",	"CHAM820101",	"KLEP840101",	"RADA880108"]
  # chem_feat = df[cols].values # (num_aa, 10)
  
  # cols = ["b2btool_backbone",	"b2btool_sidechain",	"provean"]
  # flex_feat = df[cols].values # (num_aa, 3)
  
  # cols = df.columns[df.columns.str.contains("propensity")]
  # disorder_feat = df[cols].values # (num_aa, 5)
  
  # cols = [x for x in df.keys() if 'spot_1d' in x and 'onehot' not in x]
  # spot1d_feat = df[cols].values # (num_aa, 19)
  
  # # return chem_feat, flex_feat, disorder_feat, spot1d_feat
  # all_feat = np.concatenate([chem_feat, flex_feat, disorder_feat, spot1d_feat], axis=-1)
  
  # return all_feat # (num_aa, 37)
  

### 从化学特征文件、protein序列提取该序列的化学特征
def getProtChemFeature_7s(prot_chm_df, input_complex, args):
  p_seqs = input_complex["protein_sequence"] # 字母构成的序列，字符串
  cols = ["Steric_parameter", "polarizability",	"volume",	
          "hydrophobicity",	"isoelectric point",	
          "Helix probability",	"Sheet probability"]
  all_feat = np.empty([0, 7], dtype=np.float32)
  for aa in p_seqs:
    tmp = [[prot_chm_df.loc[aa, c] for c in cols]] # (7,)
    all_feat = np.append(all_feat, tmp, axis=0)
  
  return all_feat # (num_aa, 7)
  



def computeProtChmInfo():

  cols = ["Steric_parameter", "polarizability",	"volume",	"hydrophobicity",	"isoelectric point",	
          "Helix probability",	"Sheet probability",]

  mf  = 'dataset/_datasets/cluster_res/seq_dg_v02_1.txt'
  mdf = pd.read_csv(mf, sep='\t', low_memory=False)

  keys = np.unique(mdf['protein_index'].values)

  dir = 'dataset/protein_all_feas/'

  tensor_by_cols = [np.empty([0,]) for i in range(7)]

  adict = {
    col : {x : np.empty([0,]) for x in list_aa}
    for col in cols
  }
  si = 0
  for prot_key in tqdm(keys):
    prot_key
    f = "{}.txt".format(prot_key)
    f = os.path.join(dir, f)
    df = pd.read_csv(f, sep='\t')
    
    seqs = df['protein_seq'].values
    for i, seq in enumerate(seqs):
      for col in cols:
        adict[col][seq] = np.append(adict[col][seq], [float(df.loc[i][col])], axis=0)

    if si == 100:
      break
    # chm_feat = df[cols].values # (num_aa, 7)
    # for i in range(7):
    #   tensor_by_cols[i] = np.append(tensor_by_cols[i], chm_feat[:, i], axis=0)
    si += 1
  # for tensor in tensor_by_cols:
  #   print(tensor.shape)
  #   m, s = tensor.mean(), tensor.std()
  #   print(m, s)

  for col in cols:
    print(col)
    for k, v in adict[col].items():
      print(k, v.std())



if __name__ == '__main__':
  computeProtChmInfo()
