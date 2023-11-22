import pandas as pd
import numpy as np

import os
from data_analysis.data_utils import createDir


base_dir = "../dataset/_datasets/cluster_res"
base_file = "seq_dg_v02_1.txt" # 包括了cluster等信息的表格
output_dir = "../dataset/_datasets/cluster_res/output/"
createDir(output_dir)

fpath = os.path.join(base_dir, base_file)
base_df = pd.read_csv(fpath, sep='\t')
# base_df.head(3)

uniq_key_complex = np.unique(base_df['key_complex'].values)
mask = base_df['exp_id'].isin(uniq_key_complex)
base_df = base_df[mask]
print("dataset with unique complex, dataset size = ", len(base_df))


def get_analysis_data(input_df):
  uniq_proteins = np.unique(input_df['protein_sequence'].values)
  uniq_nu_acids = np.unique(input_df['nucleotide_sequence'].values)

  prot_is_one = True if len(uniq_proteins) == 1 else False
  na_is_one = True if len(uniq_nu_acids) == 1 else False

  uniq_uniprot_ids = np.unique(input_df['UniProt'].values)
  # len(uniq_proteins), len(uniq_nu_acids), len(uniq_uniprot_ids), len(input_df)
  # (47, 3, 1, 51)

  seq2seq_id = {seq: i for i, seq in enumerate(uniq_proteins)}
  na2na_id   = {seq: i for i, seq in enumerate(uniq_nu_acids)}
  # seq_ids = [seq2seq_id[seq] for seq in input_df['protein_sequence'].values]
  # na_ids = [na2na_id[seq] for seq in input_df['nucleotide_sequence'].values]
  # print(seq_ids)
  # print(na_ids)

  prot_series = input_df['protein_sequence'].apply(lambda k : seq2seq_id[k])
  na_series   = input_df['nucleotide_sequence'].apply(lambda k : na2na_id[k])
  prot_series = prot_series.rename("pid")
  na_series = na_series.rename("nid")
  # prot_series = prot_series.astype('int32')
  # na_series = na_series.astype('int32')
  # df = pd.concat([na_series, prot_series, input_df['dG'], input_df['nucleotide_sequence'], input_df['protein_sequence']], axis=1)
  df = pd.concat([na_series, prot_series, input_df['dG'], input_df['UniProt'], input_df['nucleotide_sequence']], axis=1)
  df = df.sort_values(by=['nid', 'pid'])

  max_var = 0.
  for k, v in df.groupby("nid"):
    var = v["dG"].values.var()
    if var > max_var:
      max_var = var

  for k, v in df.groupby("pid"):
    var = v["dG"].values.var()
    if var > max_var:
      max_var = var
  

  return df, prot_is_one, na_is_one, max_var


uniq_p_cnt = 0
uniq_n_cnt = 0
sampled_cluster_cnt = 0
big_var_cnt = 0
cluster_cnt = 0
for key, grouped_df in base_df.groupby('base_class'):
  cluster_cnt += 1
  if len(grouped_df) > 10:
    
    out_df, p_one, n_one, max_var = get_analysis_data(grouped_df)
    out_df.to_csv(os.path.join(output_dir, '{}_{}.txt'.format(key, max_var)), index=False, sep='\t')
    sampled_cluster_cnt += 1
    if p_one:
      uniq_p_cnt += 1
    if n_one:
      uniq_n_cnt += 1

    if max_var > 0.5:
      big_var_cnt += 1
print(uniq_p_cnt, uniq_n_cnt, big_var_cnt, sampled_cluster_cnt)






