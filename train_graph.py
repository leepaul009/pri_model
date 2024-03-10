import os
import time

import argparse
import pickle
import zlib
from tqdm import tqdm

import numpy as np
import pandas as pd


import torch
import webdataset as wds

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from modeling.batch_convert.batch_convert import GraphBatchConvert
import utils
from modeling.graph.net_g1 import GNetVerOne






def main():
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()


  dataset_dir = "/home/paul/workspace/pri_model/dataset"
  # use default graph convert
  default_pdb_file = os.path.join(dataset_dir, "P44_relaxed_rank_002_alphafold2_ptm_model_2_seed_000.pdb")
  batch_convert = GraphBatchConvert(default_pdb_file=default_pdb_file)

  url = os.path.join(dataset_dir, "dna_intensity_dataset/example/dna-1k-{000001..000002}.tar")
  dataset = wds.WebDataset(url)
  dataloader = DataLoader(dataset, num_workers=4, batch_size=16, collate_fn=batch_convert)

  net = GNetVerOne()

  for step, data in enumerate(dataloader):
    tensor_feats, tensor_frames, tensor_indices, labels = data
    print(tensor_feats.shape, tensor_frames.shape, tensor_indices.shape, labels.shape)
    net(data)



if __name__ == '__main__':
  main()