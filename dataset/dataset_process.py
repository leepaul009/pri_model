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
  nucleotide_to_index, max_num_atoms_in_aa

from sklearn.cluster import KMeans


  
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
  
  
  
  
  
  
  
  
  

  
  