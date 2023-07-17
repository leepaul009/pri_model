import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

# set env
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 100)

### global variable:
g_nucleic_acids_types = ['dsDNA', 'dsRNA', 'ssDNA', 'ssRNA']


from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()



###########################################################################################
### 1. Process cluster
###########################################################################################


param_load = True
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
mainf = 'seq_dg_v02.txt'
wtf = 'wt_v02.tsv'


def createDir(p):
  if not os.path.exists(p):
    print("Directory {} doesn't exist, create a new.".format(p))
    os.makedirs(p)

output_dir = os.path.join(data_root, output_dir)
createDir(output_dir)

# mutation_type
# exp_id, key_complex
# protein_index, key_nucleic_acids
# protein_sequence nucleotide_sequence dG 

mainf = os.path.join(data_root, mainf)
madf = pd.read_csv(mainf, sep='\t', low_memory=False)
print(len(madf))

wtf = os.path.join(data_root, wtf)
wtdf = pd.read_csv(wtf, sep='\t', low_memory=False)
print(len(wtdf))

### check valid
# for i in tqdm(range(len(wtdf))):
#     row = wtdf.loc[i]
#     key = row['protein_sequence'] + row['nucleotide_sequence']
#     cid = row['exp_id']
#     tmp = madf[madf['exp_id'] == cid]
#     key2 = tmp['protein_sequence'].iloc[0] + tmp['nucleotide_sequence'].iloc[0]
#     if key != key2:
#         print(cid)
#     if len(tmp) != 1:
#         print(cid)


def createMap(data_root, fpath, prefix, out_mapping):
    df = pd.read_csv( os.path.join(data_root, fpath), sep='\t')
    for i in range(len(df)):
        row = df.loc[i]
        keys = row['cluster_member'].split(',')
        for k in keys:
            out_mapping[k] = prefix + row['cluster_index']


### 创建mapping
## cluster_index  cluster_number  cluster_member
## UniProt -> class_name
## key_nucleic_acids -> class_name

uniProt2cls = dict()
createMap(data_root, pwtc_f, 'P', uniProt2cls)

naKey2cls = dict()
createMap(data_root, ddna_f, 'DDNA', naKey2cls)
createMap(data_root, drna_f, 'DRNA', naKey2cls)
createMap(data_root, sdna_f, 'SDNA', naKey2cls)
createMap(data_root, srna_f, 'SRNA', naKey2cls)


def addColumnByName(df, col_name, fill_val='None', set_front=False):
  pos = 1 if set_front else len(df.columns)
  if col_name not in df.columns:
    df.insert( pos, col_name, fill_val, allow_duplicates=False )
    print('add a new column {} that does not exist'.format(col_name))
  else:
    print('column {} already exists'.format(col_name))

def createClusterColumns(df, uniProt2cls, naKey2cls, expId2wt):
  addColumnByName(df, 'wt_complex', False)
  addColumnByName(df, 'pclass', 'None')
  addColumnByName(df, 'nuclass', 'None')
  # check if input cluster file is feasible 
  uniProt_arr = np.array(list(uniProt2cls.keys()))
  except_df = df[~df['UniProt'].isin( uniProt_arr )]
  if len(except_df) > 0:
    print("following UniProtId do not exist in main data frame: {}".format(
      except_df['UniProt'].values))
  # update prot cluster into main data frame:
  df.loc[:, 'pclass'] = df['UniProt'].apply(lambda k : uniProt2cls[k])
  # update na cluster into main data frame:
  for na_type in g_nucleic_acids_types:
    ss_index = df.index[df['nucleic_acid_type_new'] == na_type]
    ss = df.loc[ss_index, 'key_nucleic_acids'].apply(lambda k : naKey2cls[k])
    df.loc[ss_index, 'nuclass'] = ss

  df.loc[:, 'wt_complex'] = df['exp_id'].apply(lambda e : expId2wt[e])

expId2wt = dict()
for c in madf.loc[:, 'exp_id'].values:
  expId2wt[c] = False
for c in wtdf.loc[:, 'exp_id'].values:
  expId2wt[c] = True

### 处理wt数据
createClusterColumns(wtdf, uniProt2cls, naKey2cls, expId2wt)
### 处理主数据
createClusterColumns(madf, uniProt2cls, naKey2cls, expId2wt)

# mutation_type
# UniProt
# nucleic_acid_type_new
# double: newnafea_na_job
# single: na_index

madf['base_class'] = madf[['nuclass', 'pclass']].agg('-'.join, axis=1)
wtdf['base_class'] = wtdf[['nuclass', 'pclass']].agg('-'.join, axis=1)


### get wt and mut sub dataframe

groups_by_wt = dict()
for k, subdf in madf.groupby('wt_complex'):
    cp = subdf.copy(deep=True)
    cp = cp.reset_index()
    print(k, len(cp))
    groups_by_wt[k] = cp
    
wt_sub_df = groups_by_wt[True]
mut_sub_df = groups_by_wt[False]

all_unique_cls = np.unique(madf['base_class'].values)
wt_cls_set = np.unique(wt_sub_df['base_class'].values)
mut_cls_set = np.unique(mut_sub_df['base_class'].values)

# in WT but not in MUT
# cls_only_in_wt = wt_cls_set - mut_cls_set
# len(all_unique_cls) - len(mut_cls_set - wt_cls_set) - len(cls_only_in_wt)
### only in WT: 3207
### only in MUT: 207
### in Both: 808
### all: 4222

print('cluster done')


######################################################################################################
### 2.1 compare similarity between protKey/NaKey
######################################################################################################

### create mapping from base class to expriment index


map_cls2size = dict()
map_cls2expIds = dict()
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

def read2df(root, f):
    f = os.path.join(root, f)
    return pd.read_csv(f, sep='\t') # , low_memory=False

dir_sim = 'similarity'

sim_prot_f = 'protein_dist_230710.txt'
sim_ddna_f = 'dsdna_dist_230710.txt'  
sim_drna_f = 'dsrna_dist_230710.txt'
sim_sdna_f = 'ssdna_dist_230710.txt'  
sim_srna_f = 'ssrna_dist_230710.txt'

sim_dir = os.path.join(data_root, dir_sim)

sim_prot_df = read2df(sim_dir, sim_prot_f)
sim_ddna_df = read2df(sim_dir, sim_ddna_f)
sim_drna_df = read2df(sim_dir, sim_drna_f)
sim_sdna_df = read2df(sim_dir, sim_sdna_f)
sim_srna_df = read2df(sim_dir, sim_srna_f)

map_naType2simDf = {
    'dsDNA' : sim_ddna_df,
    'dsRNA' : sim_drna_df,
    'ssDNA' : sim_sdna_df,
    'ssRNA' : sim_srna_df,
}


### compute similarity matrix between pKeys, between naKeys

def computeSimMat(df, sim_df, input_key, map_Key2Ind):
  '''
  inputs:
    sim_df: data frame of output from mmseq2
    input_key: prot key or na key
  outputs:
    map_Key2Ind: pKey or NaKey to sorted index
  '''
  key_set = np.unique( df[input_key].values )
  local_k2i = {nkey: lid for lid, nkey in enumerate(key_set)}
  map_Key2Ind.update(local_k2i)
      
  sim_mat = np.zeros([len(key_set), len(key_set)], dtype=np.float64)

  query_ind = sim_df['query'].apply(lambda x: local_k2i[x]).values
  target_ind = sim_df['target'].apply(lambda x: local_k2i[x]).values
  fident_arr = sim_df['fident'].values
  assert (len(query_ind) == len(target_ind))
  sim_mat[query_ind, target_ind] = fident_arr

  print( "key = {} sim_mat shape = {} query = {} target = {} similarity = {}".format(
      input_key, sim_mat.shape, len(query_ind), len(target_ind), len(fident_arr)) )
  
  return sim_mat

### process prot
map_pKey2simMatInd = dict()
prot_sim_mat = computeSimMat(madf, sim_prot_df, 'protein_index', map_pKey2simMatInd)

### process na
map_naType2simMat = dict()
map_naKey2simMatInd = dict()
for na_type, sdf in madf.groupby('nucleic_acid_type_new'):
    map_naType2simMat[na_type] = computeSimMat(sdf, 
                                               map_naType2simDf[na_type], 
                                               'key_nucleic_acids', 
                                               map_naKey2simMatInd)
print("compute similarity between protKey or naKey done")


######################################################################################################
### 2.2 compute mean similarity
######################################################################################################

### get mapping from cluster to unique sim-mat-index
map_cls_to_pSimMatInds = dict()
map_cls_to_naSimMatInds = dict()
for c, subdf in madf.groupby('base_class'):
  pmids = subdf['protein_index'].apply(lambda x : map_pKey2simMatInd[x])
  pmids = np.unique(pmids.values)
  map_cls_to_pSimMatInds[c] = pmids
  nmids = subdf['key_nucleic_acids'].apply(lambda x : map_naKey2simMatInd[x])
  nmids = np.unique(nmids.values)
  map_cls_to_naSimMatInds[c] = nmids

### get mean similarity by prot or na
def getMeanSim(sim_mat, cls_a, cls_b, cls2simMids):
  mids_a = cls2simMids[cls_a]
  mids_b = cls2simMids[cls_b]
  indices_a = np.repeat(mids_a.reshape(-1, 1), len(mids_b), axis=1) # (len_a, len_b)
  indices_b = np.repeat(mids_b.reshape(1, -1), len(mids_a), axis=0) # (len_a, len_b)
  sim = sim_mat[indices_a, indices_b] # (num_a, num_b)
  return sim.mean()

def computeDist(i, j):
  '''
  inputs:
    i, j: mean similarity of prot and na
  '''
  i += 1
  j += 1
  return i * j * (i + j) / (np.abs(i - j) + 1)


series_pMids = madf['protein_index'].apply(lambda x: map_pKey2simMatInd[x])
series_naMids = madf['key_nucleic_acids'].apply(lambda x: map_naKey2simMatInd[x])
map_expId_to_pSimId = pd.Series(series_pMids.values, index=madf['exp_id'])
map_expId_to_naSimId = pd.Series(series_naMids.values, index=madf['exp_id'])

def getNcType(class_name):
    if 'DDNACluster' in class_name:
        return 'dsDNA'
    elif 'DRNACluster' in class_name:
        return 'dsRNA'
    elif 'SDNACluster' in class_name:
        return 'ssDNA'
    elif 'SRNACluster' in class_name:
        return 'ssRNA'
    else:
        print("get base class name with a wrong NC class: {}".format(class_name))
        return None

#############################################################################
###### compute cluster inter-similarity of prot and na ######################

### map from cluster pair to their prot-similarity and na-similarity
map_cls_pair2dist = dict()

clsPair2dist_f = os.path.join(output_dir, 'cls_inter_sim_pair.npz')
if param_load and os.path.exists(clsPair2dist_f):
  clsPair2dist_npzData = np.load(clsPair2dist_f, allow_pickle=True)
  map_cls_pair2dist = clsPair2dist_npzData['map_cls_pair2dist'].item()
  print('similarity data of cluster pair exists, read: {}'.format(clsPair2dist_f))
else:
  for ci in tqdm(all_unique_cls):
    for cj in all_unique_cls:
      if ci == cj:
        map_cls_pair2dist[ci, cj] = (1.0, 1.0)
        continue
      if (ci, cj) in map_cls_pair2dist or (cj, ci) in map_cls_pair2dist:
        continue
      
      nc_type_i = getNcType(ci)
      nc_type_j = getNcType(cj)
      mP_sim = getMeanSim(prot_sim_mat, ci, cj, map_cls_to_pSimMatInds)
      mN_sim = None
      if nc_type_i == nc_type_j:
        mN_sim = getMeanSim(map_naType2simMat[nc_type_i], ci, cj, 
                            map_cls_to_naSimMatInds)
      else:
        # different type indicates largest distance 
        mN_sim = 1.0

      map_cls_pair2dist[ci, cj] = (mP_sim, mN_sim)
      map_cls_pair2dist[cj, ci] = (mP_sim, mN_sim)
  clsPair2dist_f = os.path.join(output_dir, 'cls_inter_sim_pair')
  np.savez(clsPair2dist_f, map_cls_pair2dist=map_cls_pair2dist)
  print('save similarity data of cluster pair to : {}'.format(clsPair2dist_f))

map_cls_to_mid = {c:i for i, c in enumerate(all_unique_cls)}
map_mid_to_cls = {i:c for i, c in enumerate(all_unique_cls)}


#############################################################################
###### compute cluster inter-distance #######################################

cls_dist_f = os.path.join(output_dir, 'cls_inter_dist.npz')
if param_load and os.path.exists(cls_dist_f):
  cls_dist_npz_data = np.load(cls_dist_f, allow_pickle=True)
  mDist_cls = cls_dist_npz_data['mDist_cls']
  mat_cls_dist = cls_dist_npz_data['mat_cls_dist']
  map_cls_to_mid = cls_dist_npz_data['map_cls_to_mid'].item()
  map_mid_to_cls = cls_dist_npz_data['map_mid_to_cls'].item()
  print('load existed dist file from: {}'.format(cls_dist_f))
else:
  mat_cls_dist = np.zeros([len(all_unique_cls), len(all_unique_cls)], dtype=np.float64)
  for k, v in tqdm(map_cls_pair2dist.items()):
    ci, cj = k
    ind_i = map_cls_to_mid[ci]
    ind_j = map_cls_to_mid[cj]
    if ind_i == ind_j:
      continue
    si, sj = v
    mat_cls_dist[ind_i, ind_j] = mat_cls_dist[ind_j, ind_i] = computeDist(si, sj)
  
  mDist_cls = np.mean(mat_cls_dist, axis=1)
  np.savez(os.path.join(output_dir, 'cls_inter_dist'), 
    mDist_cls=mDist_cls, 
    mat_cls_dist=mat_cls_dist,
    map_cls_to_mid=map_cls_to_mid,
    map_mid_to_cls=map_mid_to_cls)



######


table_list_cls = []
table_list_avg_dist = []
table_list_unique_eid_sz = []
table_list_same_eid_sz = []
for cls, mid in map_cls_to_mid.items():
  table_list_cls.append(cls)
  table_list_avg_dist.append(mDist_cls[mid])

  ueids = map_cls2uniqExpIds[cls]
  eids = map_cls2expIds[cls]
  table_list_unique_eid_sz.append(len(ueids))
  table_list_same_eid_sz.append(len(eids) - len(ueids))

tmp_dict = {
  'cluster' : table_list_cls,
  'avg_distance' : table_list_avg_dist,
  'size_unique_complex' : table_list_unique_eid_sz,
  'size_except_complex' : table_list_same_eid_sz,
}
cluster_df = pd.DataFrame(tmp_dict)
cluster_df.to_csv(os.path.join(output_dir, 'cluster.csv'), sep='\t')

print('compute cluster inter-distance done')

def getCandidateCls(df):
  alist = []
  mask = df['size_except_complex'] == 0
  alist.append(df[mask]['cluster'].values)
  mask = df['size_except_complex'] == 1
  alist.append(df[mask]['cluster'].values)
  return np.concatenate(alist)

candidate_cls = getCandidateCls(cluster_df)

#############################################################################
###### compute cluster inter-distance #######################################

addColumnByName(madf, 'avg_dist_to_others', 0.0)

series_tmp_indices = madf['base_class'].apply(lambda x : map_cls_to_mid[x]).values

series_avg_dist = pd.Series(mDist_cls[series_tmp_indices])
madf['avg_dist_to_others'] = series_avg_dist


if True:
  madf.to_csv(os.path.join(data_root, 'seq_dg_v02_1.txt'), index=False, sep='\t')
  wtdf.to_csv(os.path.join(data_root, 'wt_v02_1.txt'), index=False, sep='\t')


######################################################################################################
### 
######################################################################################################



def getMapEid2Uniq(df):
  map_expId2uniq = dict()
  for k, subdf in df.groupby('key_complex'):
    eids = subdf['exp_id'].values
    l = len(eids)
    for eid in eids:
      map_expId2uniq[eid] = True if l == 1 else False
  return map_expId2uniq

map_expId2uniq = getMapEid2Uniq(madf)

map_naType2df = dict()
for k, subdf in madf.groupby('nucleic_acid_type_new'):
  ### TODO: check index reset
  map_naType2df[k] = subdf.copy(deep=True)


# all_unique_cls
# mDist_cls
# map_cls_to_mid
# map_mid_to_cls
def splitData(cand_cls, df, mDistAll, cls2id, id2cls, cls2size, cls2expIds):
  '''
  df: df of this na type
  mDistAll:
  cls2id: map from cluster to dist mat index
  '''
  ### use global variable
  cur_ueids = np.unique(df['key_complex'].values)
  whole_sz = len(cur_ueids)

  cls_set = set(df['base_class'])
  chosen_mids = np.array( [cls2id[c] for c in cls_set] )
  mDist = mDistAll[chosen_mids]
  
  n_test = np.ceil(whole_sz * 0.25)
  n_train = np.ceil(whole_sz * 0.5)
  n_val = whole_sz - n_test - n_train

  sorted_indices = mDist.argsort()
  # sorted_index -> cls

  ### process test
  tmp_sum = 0
  test_cls_list = list()
  test_cls_expIds = np.empty((0), np.int64)

  for sid in sorted_indices:
    # we can use index from mDist to chosen_mids
    mid = chosen_mids[sid]
    cls = id2cls[mid]
    # if cls not in cand_cls:
    #   continue
    test_cls_list.append(cls)

    test_cls_expIds = np.append(test_cls_expIds, cls2expIds[cls], axis=0)

    sz = cls2size[cls]
    tmp_sum += sz

    if tmp_sum > n_test:
      break
  print("collect {} test items".format(tmp_sum))
  return test_cls_expIds


def splitData2(cand_cls, df, mDistAll, cls2id, id2cls, cls2size, cls2expIds):

  cur_ueids = np.unique(df['key_complex'].values)
  uniq_clss = np.unique(df['base_class'].values)
  shuf_inds = np.random.permutation(len(uniq_clss))
  
  whole_sz = len(cur_ueids)
  n_test = np.ceil(whole_sz * 0.25)

  test_cls_expIds = np.empty((0), np.int64)
  for ind in shuf_inds:
    cls = uniq_clss[ind]
    test_cls_expIds = np.append(test_cls_expIds, cls2expIds[cls], axis=0)
    if len(test_cls_expIds) > n_test:
      break
  print("collect {} test items".format(len(test_cls_expIds)))
  return test_cls_expIds

ddna_test_cls_expIds = splitData2(candidate_cls, map_naType2df['dsDNA'], mDist_cls, 
  map_cls_to_mid, map_mid_to_cls, 
  map_cls2uniqSize, map_cls2uniqExpIds)

drna_test_cls_expIds = splitData2(candidate_cls, map_naType2df['dsRNA'], mDist_cls, 
  map_cls_to_mid, map_mid_to_cls, 
  map_cls2uniqSize, map_cls2uniqExpIds)

sdna_test_cls_expIds = splitData2(candidate_cls, map_naType2df['ssDNA'], mDist_cls, 
  map_cls_to_mid, map_mid_to_cls, 
  map_cls2uniqSize, map_cls2uniqExpIds)

srna_test_cls_expIds = splitData2(candidate_cls, map_naType2df['ssRNA'], mDist_cls, 
  map_cls_to_mid, map_mid_to_cls, 
  map_cls2uniqSize, map_cls2uniqExpIds)


np.savez(os.path.join(output_dir, 'test_set_v01'),
  ddna = ddna_test_cls_expIds,
  drna = drna_test_cls_expIds,
  sdna = sdna_test_cls_expIds,
  srna = srna_test_cls_expIds)



###
test_expIds_all = [ddna_test_cls_expIds, drna_test_cls_expIds, sdna_test_cls_expIds, srna_test_cls_expIds]
test_expIds_all = np.concatenate(test_expIds_all)

map_expId2uniqExpId = dict()
for k, subdf in madf.groupby('key_complex'):
  ueid = subdf['exp_id'].values.min()
  for eid in subdf['exp_id'].values:
    map_expId2uniqExpId[eid] = ueid

uniq_test_eids = np.array(list( set([ map_expId2uniqExpId[x] for x in test_expIds_all ]) ))

new_df = madf[madf['exp_id'].isin(uniq_test_eids)]
new_df = new_df.reset_index()
print("test size = {}".format(len(new_df)))
new_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, sep='\t')


new_df = madf[~madf['exp_id'].isin(test_expIds_all)]
if True: # use uniq
  ss = new_df['exp_id'].apply(lambda x : map_expId2uniqExpId[x])
  uniq_train_eids = np.unique(ss.values)
  new_df = new_df[new_df['exp_id'].isin(uniq_train_eids)]
  new_df = new_df.reset_index()
m1 = new_df['protein_sequence'].str.len() <= 1200
m2 = new_df['nucleotide_sequence'].str.len() <= 150
new_df = new_df[m1 * m2]
new_df = new_df.reset_index()
print("train size = {}".format(len(new_df)))
new_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, sep='\t')