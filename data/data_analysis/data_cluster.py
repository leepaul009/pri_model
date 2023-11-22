import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from .data_utils import createDir, \
                        createKey2Cluster, \
                        createClusterColumns, \
                          read2df, 

# set env
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 100)

### global variable:
g_nucleic_acids_types = ['dsDNA', 'dsRNA', 'ssDNA', 'ssRNA']


# data_root = '../dataset/_datasets/cluster_res'
data_root = 'dataset/_datasets/cluster_res'
output_dir = 'out_v01'
# 分类数据
pwtc_f = 'protein_wt_cluster.tsv'
ddna_f = 'dsDNA_80_cluster.txt'
drna_f = 'dsRNA_80_cluster.txt'
sdna_f = 'ssDNA_80_cluster.txt'
srna_f = 'ssRNA_80_cluster.txt'
# 主要数据
mainf = 'seq_dg_v02.txt' # all data
wtf = 'wt_v02.tsv' # wild data


output_dir = os.path.join(data_root, output_dir)
createDir(output_dir)


mainf = os.path.join(data_root, mainf)
madf = pd.read_csv(mainf, sep='\t', low_memory=False)
print(len(madf))

wtf = os.path.join(data_root, wtf)
wtdf = pd.read_csv(wtf, sep='\t', low_memory=False)
print(len(wtdf))


uniProt2cls = dict() # protein key -> cluster name
naKey2cls   = dict() # dna/rna key -> cluster name

createKey2Cluster(data_root, pwtc_f, 'P', uniProt2cls)
createKey2Cluster(data_root, ddna_f, 'DDNA', naKey2cls)
createKey2Cluster(data_root, drna_f, 'DRNA', naKey2cls)
createKey2Cluster(data_root, sdna_f, 'SDNA', naKey2cls)
createKey2Cluster(data_root, srna_f, 'SRNA', naKey2cls)

expId2wt = dict()
for c in madf.loc[:, 'exp_id'].values:
  expId2wt[c] = False
for c in wtdf.loc[:, 'exp_id'].values:
  expId2wt[c] = True

### 处理wt数据(增加列：野生型，蛋白质cluster，核酸cluster)
createClusterColumns(wtdf, uniProt2cls, naKey2cls, expId2wt)
### 处理主数据
createClusterColumns(madf, uniProt2cls, naKey2cls, expId2wt)

madf['base_class'] = madf[['nuclass', 'pclass']].agg('-'.join, axis=1)
wtdf['base_class'] = wtdf[['nuclass', 'pclass']].agg('-'.join, axis=1)

######################################################################################################
### 2.1 compare similarity between protKey/NaKey
######################################################################################################

map_cls2size = dict()
map_cls2expIds = dict() # 实验id
map_cls2uniqSize = dict()
map_cls2uniqExpIds = dict()
g_overall_uniqSize = 0
for k, subdf in madf.groupby('base_class'):
  map_cls2size[k] = len(subdf)
  map_cls2expIds[k] = subdf['exp_id'].values
  ### here uniq indicates unique complex
  uniq_exp_ids = np.unique(subdf['key_complex'].values)
  map_cls2uniqExpIds[k] = uniq_exp_ids
  map_cls2uniqSize[k] = len(uniq_exp_ids)
  g_overall_uniqSize += len(uniq_exp_ids)
print("g_overall_uniqSize = {}".format(g_overall_uniqSize))



sim_dir = os.path.join(data_root, 'similarity')
sim_prot_f = 'protein_dist_230710.txt'
sim_ddna_f = 'dsdna_dist_230710.txt'  
sim_drna_f = 'dsrna_dist_230710.txt'
sim_sdna_f = 'ssdna_dist_230710.txt'  
sim_srna_f = 'ssrna_dist_230710.txt'

sim_prot_df = read2df(sim_dir, sim_prot_f)
sim_ddna_df = read2df(sim_dir, sim_ddna_f)
sim_drna_df = read2df(sim_dir, sim_drna_f)
sim_sdna_df = read2df(sim_dir, sim_sdna_f)
sim_srna_df = read2df(sim_dir, sim_srna_f)


