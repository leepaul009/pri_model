import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
  
def createDir(p):
  if not os.path.exists(p):
    print("Directory {} doesn't exist, create a new.".format(p))
    os.makedirs(p)

def read2df(root, f):
    f = os.path.join(root, f)
    return pd.read_csv(f, sep='\t') # , low_memory=False

def createKey2Cluster(file_dir, file_name, prefix, out_mapping):
    df = pd.read_csv( os.path.join(file_dir, file_name), sep='\t')
    for i in range(len(df)):
        row = df.loc[i]
        keys = row['cluster_member'].split(',')
        for k in keys:
            out_mapping[k] = prefix + row['cluster_index']


# insert a column(named col_name) into dataframe
# with default value(default_value)
def addColumnByName(df, col_name, default_value='None', set_front=False):
  pos = 1 if set_front else len(df.columns)
  if col_name not in df.columns:
    df.insert( pos, col_name, default_value, allow_duplicates=False )
    print('add a new column {} that does not exist'.format(col_name))
  else:
    print('column {} already exists'.format(col_name))


def createClusterColumns(df, uniProt2cls, naKey2cls, expId2wt):
  addColumnByName(df, 'wt_complex', False)
  addColumnByName(df, 'pclass', 'None')
  addColumnByName(df, 'nuclass', 'None')
  
  # check if input cluster file is feasible 
  uniq_prot_keys = np.array(list(uniProt2cls.keys()))
  except_df = df[~df['UniProt'].isin( uniq_prot_keys )]
  if len(except_df) > 0:
    print("following UniProtId in cluster file do not exist in main data frame: {}".format(
      except_df['UniProt'].values))
  
  # update prot cluster into main data frame:
  df.loc[:, 'pclass'] = df['UniProt'].apply(lambda k : uniProt2cls[k])
  
  # update na cluster into main data frame:
  # for na_type in g_nucleic_acids_types:
  #   ss_index = df.index[df['nucleic_acid_type_new'] == na_type]
  #   ss = df.loc[ss_index, 'key_nucleic_acids'].apply(lambda k : naKey2cls[k])
  #   df.loc[ss_index, 'nuclass'] = ss
  
  uniq_nu_keys = np.array(list(naKey2cls.keys()))
  except_df = df[~df['key_nucleic_acids'].isin( uniq_nu_keys )]
  if len(except_df) > 0:
    print("following dna/rna key in cluster file do not exist in main data frame: {}".format(
      except_df['key_nucleic_acids'].values))
  
  df.loc[:, 'nuclass'] = df['key_nucleic_acids'].apply(lambda k : naKey2cls[k])

  df.loc[:, 'wt_complex'] = df['exp_id'].apply(lambda e : expId2wt[e])
  
  
  
def computeSimMat(df, sim_df, input_key, map_Key2Ind):
  '''
  inputs:
    sim_df: data frame of output from mmseq2
    input_key: prot key or na key
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
  
  return sim_mat, map_Key2Ind