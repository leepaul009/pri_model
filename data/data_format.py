import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', 100)

### global variable:
g_nucleic_acids_types = ['dsDNA', 'dsRNA', 'ssDNA', 'ssRNA']




# UniProt
# nucleic_acid_type_new
# double: newnafea_na_job
# single: na_index
# The "nucleic_acid_type_new" column represents the four types of nucleic acids. 
# The "dG_ori" column contains the original dG values obtained from the literature. 
# If value of "complex_type" column is "dele,", indicates row needs to be removed.

def addColumnByName(df, col_name, set_front=False, set_str=True):
  pos = 1 if set_front else len(df.columns)
  fill_val = 'None' if set_str else -1
  if col_name not in df.columns:
    df.insert( pos, col_name, fill_val, allow_duplicates=False )
    print('add a new column {} that does not exist'.format(col_name))
  else:
    print('column {} already exists'.format(col_name))

def updateColumnNaType(df):
  col_name = 'key_nucleic_acids'
  addColumnByName(df, col_name)
  # for i in range(len(df)):
  #   row = df.loc[i]
  #   na_type = row['nucleic_acid_type_new']
  #   if na_type == 'dsDNA':
  #       df.loc[i, col_name] = row['newnafea_na_job'].replace('new_DNA', 'Double_DNA')
  #   elif na_type == 'dsRNA':
  #       df.loc[i, col_name] = row['newnafea_na_job'].replace('new_RNA', 'Double_RNA')
  #   elif na_type == 'ssDNA':
  #       df.loc[i, col_name] = row['na_index'].replace('ssDNA', 'Single_DNA_')
  #   elif na_type == 'ssRNA':
  #       df.loc[i, col_name] = row['na_index'].replace('ssRNA', 'Single_RNA_')

  sst = df['nucleic_acid_type_new'] == 'dsDNA'
  ss = df[sst]['newnafea_na_job'].str.replace('new_DNA', 'Double_DNA')
  df.loc[ss.index, 'key_nucleic_acids'] = ss

  sst = df['nucleic_acid_type_new'] == 'dsRNA'
  ss = df[sst]['newnafea_na_job'].str.replace('new_RNA', 'Double_RNA')
  df.loc[ss.index, 'key_nucleic_acids'] = ss

  sst = df['nucleic_acid_type_new'] == 'ssDNA'
  ss = df[sst]['na_index'].str.replace('ssDNA', 'Single_DNA_')
  df.loc[ss.index, 'key_nucleic_acids'] = ss

  sst = df['nucleic_acid_type_new'] == 'ssRNA'
  ss = df[sst]['na_index'].str.replace('ssRNA', 'Single_RNA_')
  df.loc[ss.index, 'key_nucleic_acids'] = ss

### suppose exp_id existed
def updateColumnComplex(df):
  col_name = 'key_complex'
  addColumnByName(df, col_name, set_front=True, set_str=False)
  curr_id = 0
  for k, subdf in df.groupby(['protein_sequence', 'nucleotide_sequence', 'nucleic_acid_type_new']):
    ss = subdf['exp_id']
    df.loc[ss.index, col_name] = int( ss.values.min() )
    curr_id += 1
  print("totally have {} unique complex".format(curr_id))



data_root = 'dataset/_datasets/cluster_res/'
### from_files located in orig_root
orig_root = 'original_files'
### from files
main_f = 'seq_dg_230703.txt'
wt_f = 'wt.tsv'
### save to files
new_main_f = 'seq_dg_v02.txt'
new_wt_f = 'wt_v02.tsv'

main_f = os.path.join(data_root, orig_root, main_f)
main_df = pd.read_csv(main_f, sep='\t', low_memory=False)

wt_f = os.path.join(data_root, orig_root, wt_f)
wt_df = pd.read_csv(wt_f, sep='\t', low_memory=False)

### delete some rows
main_df = main_df[main_df['complex_type'] != 'dele'].reset_index(drop=True)

### update uniProtId
main_df['UniProt'] = main_df['UniProt'].apply(lambda x: x.replace('A0A140NGK1','P0A6X3'))
main_df['UniProt'] = main_df['UniProt'].apply(lambda x: x.replace('C1IFD2','P0A6X3'))
main_df['UniProt'] = main_df['UniProt'].apply(lambda x: x.replace('Q71TA8','P24042'))
main_df['UniProt'] = main_df['UniProt'].apply(lambda x: x.replace('A0A0H2UPA7','Q8DQG2'))
main_df['UniProt'] = main_df['UniProt'].apply(lambda x: x.replace('C3SZN7','P0AGK8'))

wt_df['UniProt'] = wt_df['UniProt'].apply(lambda x: x.replace('A0A140NGK1','P0A6X3'))
wt_df['UniProt'] = wt_df['UniProt'].apply(lambda x: x.replace('C1IFD2','P0A6X3'))
wt_df['UniProt'] = wt_df['UniProt'].apply(lambda x: x.replace('Q71TA8','P24042'))
wt_df['UniProt'] = wt_df['UniProt'].apply(lambda x: x.replace('A0A0H2UPA7','Q8DQG2'))


### change col name: complex id -> exp id
main_df.rename(columns={'complex_id': 'exp_id'}, inplace=True)
wt_df.rename(columns={'complex_id': 'exp_id'}, inplace=True)

### add column of NA key
updateColumnNaType(main_df)
updateColumnNaType(wt_df)

### add complex key with min exp_id of rows belong to same complex
updateColumnComplex(main_df)
updateColumnComplex(wt_df)

new_main_f = os.path.join(data_root, new_main_f)
new_wt_f = os.path.join(data_root, new_wt_f)
main_df.to_csv(new_main_f, sep='\t', index=False)
wt_df.to_csv(new_wt_f, sep='\t', index=False)
print('save updated dataframe to {}'.format(new_main_f))
print('save to {}'.format(new_wt_f))



# newnafea_na_job：用于索引 2个双链cluster文件 和 2个双链相似度文件
# na_index： 用于索引 2个单链cluster文件 和 2个单链相似度文件
# nucleic_acid_type_new: 核算的种类

def read2df(root, f):
    f = os.path.join(root, f)
    return pd.read_csv(f, sep='\t') # , low_memory=False

def write2csv(root, f, df):
    f = os.path.join(root, f)
    df.to_csv(f, sep='\t', index=False)
    print("write to {}".format(f))
    
pwtc_f = 'protein_wt_cluster.tsv'
ddna_f = 'dsDNA_80_cluster.txt'
drna_f = 'dsRNA_80_cluster.txt'
sdna_f = 'ssDNA_80_cluster.txt'
srna_f = 'ssRNA_80_cluster.txt'

pwtc_df = read2df(data_root, pwtc_f)
ddna_df = read2df(data_root, ddna_f)
drna_df = read2df(data_root, drna_f)
sdna_df = read2df(data_root, sdna_f)
srna_df = read2df(data_root, srna_f)

def convertD(df, cols):
    return df.apply(lambda x: x.replace(
                {'new_DNA': 'Double_DNA', 
                 'new_RNA': 'Double_RNA',
                 'ssDNA': 'Single_DNA_',
                 'ssRNA': 'Single_RNA_'
                },
                regex=True))

changed_cols = ['protein_index', 'cluster_member']

ddna_df = convertD(ddna_df, changed_cols)
drna_df = convertD(drna_df, changed_cols)
sdna_df = convertD(sdna_df, changed_cols)
srna_df = convertD(srna_df, changed_cols)


write2csv(data_root, ddna_f, ddna_df)
write2csv(data_root, drna_f, drna_df)
write2csv(data_root, sdna_f, sdna_df)
write2csv(data_root, srna_f, srna_df)


# fident表示距离,范围在0-1之间; 
# 相同的组合,例如 protein_6 和 protein_12之间的dist是相同的,
# 若在核酸的输出文件中检索不到两两之间的相似性，请用0填充,说明其相似性很低;
dir_sim = 'similarity'

sim_ddna_f = 'dsdna_dist_230703.txt'  
sim_drna_f = 'dsrna_dist_230703.txt'
sim_sdna_f = 'ssdna_dist_230703.txt'  
sim_srna_f = 'ssrna_dist_230703.txt'

sim_dir = os.path.join(data_root, dir_sim)


sim_ddna_df = read2df(sim_dir, sim_ddna_f)
sim_drna_df = read2df(sim_dir, sim_drna_f)
sim_sdna_df = read2df(sim_dir, sim_sdna_f)
sim_srna_df = read2df(sim_dir, sim_srna_f)

changed_cols = ['query', 'target']
sim_ddna_df = convertD(sim_ddna_df, changed_cols)
sim_drna_df = convertD(sim_drna_df, changed_cols)
sim_sdna_df = convertD(sim_sdna_df, changed_cols)
sim_srna_df = convertD(sim_srna_df, changed_cols)

write2csv(sim_dir, sim_ddna_f, sim_ddna_df)
write2csv(sim_dir, sim_drna_f, sim_drna_df)
write2csv(sim_dir, sim_sdna_f, sim_sdna_df)
write2csv(sim_dir, sim_srna_f, sim_srna_df)

print('done')














#230703 
#双链核酸的相似性文件已经修改,是格式的问题,现在能够直接按照 na_index 列进行索引;
#下面是例子;
#dsdnaSim = pd.read_table('/data/sswwkk/Protein_sequence/quanzhe_share/absolute_affinity/date/230702/sim_matrix/dsDNA_ident_matrix.txt',index_col=0)
#seq_dg = pd.read_csv('/data/sswwkk/Protein_sequence/quanzhe_share/absolute_affinity/date/230702/seq_dg_230702.txt',sep='\t')
#seq_dg_dsDNA = seq_dg[seq_dg['nucleic_acid_type_new']=='dsDNA']
#通过以下方式索引
#dsdnaSim.loc['dsdnanew_DNA_3312','dsdnanew_DNA_2549']