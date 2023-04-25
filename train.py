import os
import numpy as np
import pandas as pd
import argparse
import tqdm

from preprocessing import PDBio
from preprocessing import pipelines
from preprocessing import PDB_processing
from utilities.paths import structures_folder
import utilities.dataset_utils as dataset_utils

import torch

from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict
import utils
from dataset_pri import PriDataset

def main2():

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



def save_ckpt(model, opt, save_dir, epoch, iter):
  if not os.path.exists(save_dir):
    print("Directory {} doesn't exist, create a new.".format(save_dir))
    os.makedirs(save_dir)

  # process model for multi-GPU or single-GPU case:
  model_to_save = model.module if hasattr(
    model, 'module') else model  # Only save the model it-self

  # move from gpu to cpu
  state_dict = model_to_save.state_dict()
  for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()

  output_model_file = os.path.join(
    save_dir, "model.{}.{}.ckpt".format(epoch + 1, iter))
  torch.save(
    {"epoch": epoch + 1, 
      "state_dict": state_dict, 
      "opt_state": opt.state_dict()},
    output_model_file)


def val(model, device, args):

  dataset = PriDataset(args, args.eval_batch_size)
  # sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank())
  sampler = SequentialSampler(dataset)
  dataloader = torch.utils.data.DataLoader(
    dataset, sampler=sampler,
    batch_size=args.eval_batch_size,
    collate_fn=utils.batch_list_to_batch_tensors,
    pin_memory=False)


  # model.to(device)
  model.eval()

  iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
  for step, batch in enumerate(iter_bar):
    with torch.no_grad():
      loss, pred, _ = model(batch, device)
      # print eval info

  # for training of next epoch
  model.train()


def train_one_epoch(model, train_dataloader, optimizer, device, i_epoch, args):
  
  save_iters = 1000
  save_dir = ''

  for step, batch in enumerate(train_dataloader):
    print("step {}, batch.type={}".format( step, type(batch) ))
    if (isinstance(batch, list)):
      # batch has 64 inputs(list of dict)
      print("batch size={}".format( len(batch) ))

    # break when meeting max iter

    loss, DE, _ = model(batch, device)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if is_main_process and step % save_iters == 0:
      save_ckpt(model, optimizer, save_dir, i_epoch, step)

  # do eval after an epoch training
  # val(model, device，args)


def main():
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()


  torch.cuda.set_device(args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  train_dataset = PriDataset(args, args.train_batch_size)
  train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler,
    batch_size=args.train_batch_size // get_world_size(),
    collate_fn=utils.batch_list_to_batch_tensors)
  



if __name__ == '__main__':
  ###
  main()






