import os
import numpy as np
import pandas as pd
import argparse
import tqdm
from tqdm import tqdm

from preprocessing import PDBio
from preprocessing import pipelines
from preprocessing import PDB_processing
from utilities.paths import structures_folder
import utilities.dataset_utils as dataset_utils

import torch

from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict
import utils
from dataset.dataset_pri import PriDataset

from modeling.graphnet import GraphNet

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



def learning_rate_decay(args, i_epoch, optimizer, optimizer_2=None):
  # utils.i_epoch = i_epoch

  if 'set_predict' in args.other_params:
      if not hasattr(args, 'set_predict_lr'):
          args.set_predict_lr = 1.0
      else:
          args.set_predict_lr *= 0.9

      if i_epoch > 0 and i_epoch % 5 == 0:
          for p in optimizer.param_groups:
              p['lr'] *= 0.3

      if 'complete_traj-3' in args.other_params:
          assert False

  else:
      if i_epoch > 0 and i_epoch % 5 == 0:
          for p in optimizer.param_groups:
              p['lr'] *= 0.3

      if 'complete_traj-3' in args.other_params:
          if i_epoch > 0 and i_epoch % 5 == 0:
              for p in optimizer_2.param_groups:
                  p['lr'] *= 0.3

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


def val(model, dataloader, epoch, device, args):

  # model.to(device)
  model.eval()

  iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
  for step, batch in enumerate(iter_bar):
    with torch.no_grad():
      loss, pred, _ = model(batch, device)

  print("validation result: epoch={} loss={}".format(epoch, loss.item()))

  # for training of next epoch
  model.train()


def train_one_epoch(model, train_dataloader, val_dataloader, 
                    optimizer, device, i_epoch, args):
  
  save_iters = 1000
  save_dir = args.output_dir

  for step, batch in enumerate(train_dataloader):
    # print("step {}, batch.type={}".format( step, type(batch) ))
    # if (isinstance(batch, list)):
    #   # batch has 64 inputs(list of dict)
    #   print("batch size={}".format( len(batch) ))

    # break when meeting max iter

    loss, DE, _ = model(batch, device)
    # del DE
    loss = loss.type(torch.float32)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if is_main_process and step % 10 == 0:
      print("epoch={} step={} loss={}".format(i_epoch, step, loss.item()))
    if is_main_process and step % save_iters == 0:
      save_ckpt(model, optimizer, save_dir, i_epoch, step)

  # do eval after an epoch training
  val(model, val_dataloader, i_epoch, device, args)


def main():
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()


  torch.cuda.set_device(args.local_rank)
  device = torch.device('cuda', args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  train_dataset = PriDataset(args, args.data_dir, args.train_batch_size)
  train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
  train_dataloader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler,
    batch_size=args.train_batch_size // get_world_size(),
    collate_fn=utils.batch_list_to_batch_tensors)
  

  val_dataset = PriDataset(args, args.data_dir_for_val, args.eval_batch_size)
  # sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank())
  val_sampler = SequentialSampler(val_dataset)
  val_dataloader = torch.utils.data.DataLoader(
    val_dataset, sampler=val_sampler,
    batch_size=args.eval_batch_size,
    collate_fn=utils.batch_list_to_batch_tensors,
    pin_memory=False)


  config = dict()
  model = GraphNet(config, args)
  model = model.cuda()


  if distributed:
    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True,
        device_ids=[args.local_rank], 
        output_device=args.local_rank)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  for i_epoch in range(int(args.num_train_epochs)):

    learning_rate_decay(args, i_epoch, optimizer)
    # get_rank
    if is_main_process():
      print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
      print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])

    # TODO: more function
    train_sampler.set_epoch(i_epoch)
    
    train_one_epoch(model, train_dataloader, val_dataloader, optimizer, device, i_epoch, args)


if __name__ == '__main__':
  ###
  main()






