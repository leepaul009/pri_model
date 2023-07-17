


### columns of main csv
```
exp_id: index of experiment result, one complex could have several experiment results, each result have unique exp_id

key_complex: unique complex index, two same complex have same key_complex

nucleic_acid_type_new: type of nucleic_acid

protein_index: a unique index of protein contained in this complex. used to query protein cluster files and protein-inter similarity files.
key_nucleic_acids: a unique index of nucleic_acids contained in this complex. sed to query nucleic_acids cluster files and nucleic_acids-inter similarity files.

wt_complex: true if complex is wt type

pclass: cluster name of protein
nuclass: cluster name of nucleic_acid
base_class: cluster name of complex
avg_dist_to_others: average distance to all other complex-clusters

```

### structure
```
├── cluster (cluster files, use column protein_index and key_nucleic_acids to query)
│   ├── dsDNA_80_cluster.txt
│   ├── dsRNA_80_cluster.txt
│   ├── ssDNA_80_cluster.txt
│   └── ssRNA_80_cluster.txt
├── out_v01
│   ├── cls_inter_dist.npz (cluster-to-cluster distance)
│   ├── cls_inter_sim_pair.npz 
│   ├── cluster.csv
│   ├── test.csv (test set)
│   ├── test_set_v01.npz
│   └── train.csv (trainning set)
├── readme.md
├── seq_dg_v02_1.txt (main csv file)
└── similarity (similarity files, use column protein_index and key_nucleic_acids to query)
    ├── dsdna_dist_230703.txt
    ├── dsrna_dist_230703.txt
    ├── protein_dist_230703.txt
    ├── ssdna_dist_230703.txt
    └── ssrna_dist_230703.txt


inside cls_inter_dist file, there're four items:
  mDist_cls: np.array (num_clusters) mean distance of one cluster to all other clusters
  mat_cls_dist: np.array (num_clusters, num_clusters) distance of two clusters
  map_cls_to_mid: dict, map from cluster name to matrix_index, where matrix_index is the index of mDist_cls and mat_cls_dist
  map_mid_to_cls: dict, map from matrix_index to cluster name

## use following codes to get:

data = np.load('cls_inter_dist.npz', allow_pickle=True)
mDist_cls = data['mDist_cls']
mat_cls_dist = data['mat_cls_dist']
map_cls_to_mid = data['mat_cls_dist'].item()
map_mid_to_cls = data['mat_cls_dist'].item()

```

