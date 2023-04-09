import os
import numpy as np
import pandas as pd
import argparse
from preprocessing import PDBio
from preprocessing import pipelines
from preprocessing import PDB_processing
from utilities.paths import structures_folder
import utilities.dataset_utils as dataset_utils


if __name__ == '__main__':

  list_datasets = [
    'train',
    'validation_70',
    'validation_homology',
    'validation_topology',
    'validation_none',
    'test_70',
    'test_homology',
    'test_topology',
    'test_none',
    ]

  list_dataset_names = [
    'Train',
    'Validation (70\%)',
    'Validation (Homology)',
    'Validation (Topology)',
    'Validation (None)',
    'Test (70\%)',
    'Test (Homology)',
    'Test (Topology)',
    'Test (None)'
    ]

  pipeline = pipelines.ScanNetPipeline(
    with_atom=True,
    aa_features='sequence',
  )

  list_dataset_locations = ['datasets/PPBS/labels_%s.txt'% dataset 
    for dataset in list_datasets]
  dataset_table = pd.read_csv('datasets/PPBS/table.csv',sep=',')

  list_inputs = []
  list_outputs = []
  list_weights = []

  for dataset,dataset_name,dataset_location in zip(
    list_datasets,list_dataset_names,list_dataset_locations):
    
    # List of residue-wise labels
    (list_origins,# List of chain identifiers (e.g. [1a3x_A,10gs_B,...])
    list_sequences,# List of corresponding sequences.
    list_resids,#List of corresponding residue identifiers.
    list_labels)  = dataset_utils.read_labels(dataset_location)












