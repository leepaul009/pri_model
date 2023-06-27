import os
import numpy as np
import pandas as pd
import argparse
import tqdm
from tqdm import tqdm
import time

from preprocessing import PDBio
from preprocessing import pipelines
from preprocessing import PDB_processing
from utilities.paths import structures_folder
import utilities.dataset_utils as dataset_utils

import torch

from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utilities.comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict
import utils
from dataset.dataset_pri import PriDataset, PriDatasetExt

from modeling.graphnet import GraphNet, PostProcess

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 


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

  epoch_update_lr = 100
  if i_epoch > 0 and i_epoch % epoch_update_lr == 0:
    for p in optimizer.param_groups:
      p['lr'] *= 0.75

  # if 'set_predict' in args.other_params:
  #     if not hasattr(args, 'set_predict_lr'):
  #         args.set_predict_lr = 1.0
  #     else:
  #         args.set_predict_lr *= 0.9
  #     if i_epoch > 0 and i_epoch % 5 == 0:
  #         for p in optimizer.param_groups:
  #             p['lr'] *= 0.3
  #     if 'complete_traj-3' in args.other_params:
  #         assert False
  # else:
  #     if i_epoch > 0 and i_epoch % 5 == 0:
  #         for p in optimizer.param_groups:
  #             p['lr'] *= 0.3

  #     if 'complete_traj-3' in args.other_params:
  #         if i_epoch > 0 and i_epoch % 5 == 0:
  #             for p in optimizer_2.param_groups:
  #                 p['lr'] *= 0.3

def save_ckpt(model, opt, save_dir, epoch, iter):
  save_dir = os.path.join(save_dir, "checkpoint")
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
  print("save checkpoint to {}".format(output_model_file))
  torch.save(
    {"epoch": epoch + 1, 
      "state_dict": state_dict, 
      "opt_state": opt.state_dict()},
    output_model_file)


def test(model, dataloader, post_process, epoch, device, args):
  model.eval()
  iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
  metrics = dict()
  for step, batch in enumerate(iter_bar):
    with torch.no_grad():
      loss, pred, loss_output = model(batch, device)
      post_out = post_process(loss_output)
      post_process.append(metrics, post_out, pred, batch)
  post_process.display(metrics, epoch)


def val(model, dataloader, post_process, epoch, device, args):

  # model.to(device)
  model.eval()

  iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
  metrics = dict()
  
  for step, batch in enumerate(iter_bar):
    with torch.no_grad():
      loss, pred, loss_output = model(batch, device)
      post_out = post_process(loss_output)
      post_process.append(metrics, post_out, pred, batch)
  
  post_process.display(metrics, epoch)
  # mean_loss = 0
  # print("validation result: epoch={} loss={}".format(epoch, mean_loss))

  # for training of next epoch
  model.train()


def train_one_epoch(model, train_dataloader, val_dataloader, 
                    optimizer, post_process, device, i_epoch, args):
  
  save_iters = 1000
  save_dir = args.output_dir
  metrics = dict()
  
  for step, batch in enumerate(train_dataloader):
    # print("step {}, batch.type={}".format( step, type(batch) ))
    # if (isinstance(batch, list)):
    #   # batch has 64 inputs(list of dict)
    #   print("batch size={}".format( len(batch) ))

    # break when meeting max iter

    start_time = time.time()

    loss, pred, loss_output = model(batch, device)
    post_out = post_process(loss_output)
    post_process.append(metrics, post_out)
    # del DE
    loss = loss.type(torch.float32)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if is_main_process and step % 10 == 0:
      # print("epoch={} step={} loss={}".format(i_epoch, step, loss.item()))
      end_time = time.time()
      post_process.display(metrics, i_epoch, step, args.learning_rate, end_time - start_time)
      
    # if is_main_process and step % save_iters == 0:
    #   save_ckpt(model, optimizer, save_dir, i_epoch, step)

  # do eval after an epoch training
  if args.do_eval:
    val(model, val_dataloader, post_process, i_epoch, device, args)

  if is_main_process:
    save_ckpt(model, optimizer, save_dir, i_epoch, step)

def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)

def preprocess(args):
  save_dir = os.path.join("output", args.output_dir)
  if not os.path.exists(save_dir):
    print("Directory {} doesn't exist, create a new.".format(save_dir))
    os.makedirs(save_dir)

def main():
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()
  preprocess(args)

  torch.cuda.set_device(args.local_rank)
  device = torch.device('cuda', args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  config = dict()
  model = GraphNet(config, args)
  model = model.cuda()

  post_process = PostProcess(args.output_dir)
    
  if distributed:
    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True,
        device_ids=[args.local_rank], 
        output_device=args.local_rank)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  start_epoch = 0
  if args.resume:
    ckpt_path = args.resume_path
    # if not os.path.isabs(ckpt_path):
    #     ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt["state_dict"])
    start_epoch = ckpt["epoch"] + 1
    optimizer.load_state_dict(ckpt["opt_state"])
    print("load ckpt from {} and train from epoch {}".format(ckpt_path, start_epoch))



  
  if args.do_test:
    if args.data_name == 'hox_data':
      test_dataset = PriDatasetExt(args, args.data_dir_for_test, args.test_batch_size)
    else:
      test_dataset = PriDataset(args, args.data_dir_for_test, args.test_batch_size)
    # sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank())
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
      test_dataset, sampler=test_sampler,
      batch_size=args.test_batch_size,
      collate_fn=utils.batch_list_to_batch_tensors,
      pin_memory=False)

    test(model, test_dataloader, post_process, start_epoch, device, args)

  else:
    if args.data_name == 'hox_data':
      train_dataset = PriDatasetExt(args, args.data_dir, args.train_batch_size)
    else:
      train_dataset = PriDataset(args, args.data_dir, args.train_batch_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    train_dataloader = torch.utils.data.DataLoader(
      train_dataset, sampler=train_sampler,
      batch_size=args.train_batch_size // get_world_size(),
      collate_fn=utils.batch_list_to_batch_tensors)
    
    if args.data_name == 'hox_data':
      val_dataset = PriDatasetExt(args, args.data_dir_for_val, args.eval_batch_size)
    else:
      val_dataset = PriDataset(args, args.data_dir_for_val, args.eval_batch_size)
    # sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank())
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
      val_dataset, sampler=val_sampler,
      batch_size=args.eval_batch_size,
      collate_fn=utils.batch_list_to_batch_tensors,
      pin_memory=False)

    for i_epoch in range(int(start_epoch), int(start_epoch + args.num_train_epochs)):

      learning_rate_decay(args, i_epoch, optimizer)
      # get_rank
      if is_main_process():
        print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
        print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])

      # TODO: more function
      train_sampler.set_epoch(i_epoch)
      
      train_one_epoch(model, train_dataloader, val_dataloader, optimizer, post_process, device, i_epoch, args)


if __name__ == '__main__':
  ###
  main()






