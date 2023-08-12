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

from data.dataset_pri import PriDataset, PriDatasetExt
from data.distributed_sampler import RepeatFactorTrainingSampler
from utilities.schedular import WarmupExponentialLR, WarmupCosineAnnealingLR

from modeling.globalnet import GlobalNet
from modeling.post_process import PostProcess
from modeling.esm.data import Alphabet
from modeling.dbert.data import Alphabet as DAlphabet
from modeling.batch_convert.batch_convert import BasicBatchConvert


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

from train import lr_decay_by_steps, learning_rate_decay, save_ckpt, \
  load_pretrain, preprocess


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
                    optimizer,
                    post_process, device, i_epoch, args):
  save_iters = 1000
  save_dir = args.output_dir
  metrics = dict()

  steps_sz = len(train_dataloader)

  for step, batch in enumerate(train_dataloader):
    start_time = time.time()

    loss, pred, loss_output = model(batch, device)
    post_out = post_process(loss_output)
    post_process.append(metrics, post_out)
    loss = loss.type(torch.float32)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if is_main_process and step % args.display_steps == 0:
      end_time = time.time()
      curr_lr = optimizer.param_groups[0]['lr']
      post_process.display(metrics, i_epoch, step, curr_lr, end_time - start_time)
      
    if is_main_process and step > 5000 and step % save_iters == 0:
      save_ckpt(model, optimizer, save_dir, i_epoch, step)
    
    if args.step_lr:
      lr_decay_by_steps(args, steps_sz, step, optimizer)

  # do eval after an epoch training
  if args.do_eval:
    val(model, val_dataloader, post_process, i_epoch, device, args)

  if is_main_process:
    save_ckpt(model, optimizer, save_dir, i_epoch, step)

def main():
  parser = argparse.ArgumentParser()
  utils.add_argument(parser)
  args: utils.Args = parser.parse_args()
  preprocess(args)

  if args.use_cpu:
    device = torch.device('cpu')
  else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)

  distributed = False
  if distributed:
      torch.distributed.init_process_group(backend="nccl", init_method="env://",)


  path_esm2_cp = "dataset/checkpoints/esm2_t6_8M_UR50D.pt"
  checkpoint = torch.load(path_esm2_cp)
  cfg = checkpoint['cfg_model']
  esm2_state = checkpoint['model']
  # esm2_state.update(checkpoint["regression"])
  
  path_dbm_cp = 'dataset/checkpoints/dnabert_t12.pt'
  checkpoint = torch.load(path_dbm_cp)
  dbm_state = checkpoint['model']

  alphabet = Alphabet.from_architecture("ESM-1b")
  dalphabet = DAlphabet.from_architecture()
  basic_batch_convert = BasicBatchConvert(alphabet, dalphabet)
  model = GlobalNet(args, cfg, alphabet)

  def update_state_dict(state_dict):
    state_dict = {'esm2.' + name : param for name, param in state_dict.items()}
    return state_dict
  
  def update_state_dict_dbm(state_dict):
    state_dict = {'dbm.' + name : param for name, param in state_dict.items()}
    return state_dict

  ### only load esm2 checkpoint
  esm2_state = update_state_dict(esm2_state)
  dbm_state = update_state_dict_dbm(dbm_state)
  esm2_state.update(dbm_state)
  
  model.load_state_dict(esm2_state, strict=False)

  model = model.to(device)
  
  post_process = PostProcess(args.output_dir)
    
  if distributed:
    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True,
        device_ids=[args.local_rank], 
        output_device=args.local_rank)

  optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate,
    betas=(0.9, 0.98), weight_decay=0.01)

  start_epoch = 0
  if args.resume:
    ckpt_path = args.resume_path
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt["state_dict"])
    start_epoch = ckpt["epoch"] + 1
    optimizer.load_state_dict(ckpt["opt_state"])
    print("load ckpt from {} and train from epoch {}".format(ckpt_path, start_epoch))


  # use 20 epoch from init_lr to 0
  # lr_scheduler = WarmupCosineAnnealingLR(
  #   optimizer, 
  #   T_max=args.num_train_epochs - args.warmup_epoch, 
  #   warmup_epochs=args.warmup_epoch)
  lr_scheduler = WarmupExponentialLR(
    optimizer, gamma=0.95, warmup_epochs=args.warmup_epoch)
  
  
  if args.do_test:
    if args.data_name == 'hox_data':
      test_dataset = PriDatasetExt(args, args.data_dir_for_test, args.test_batch_size)
    else:
      test_dataset = PriDataset(args, args.data_dir_for_test, args.test_batch_size)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
      test_dataset, sampler=test_sampler,
      batch_size=args.test_batch_size,
      # collate_fn=utils.batch_list_to_batch_tensors,
      collate_fn=basic_batch_convert,
      pin_memory=False)

    test(model, test_dataloader, post_process, start_epoch, device, args)
  else:
    if args.data_name == 'hox_data':
      train_dataset = PriDatasetExt(args, args.data_dir, args.train_batch_size)
    else:
      train_dataset = PriDataset(args, args.data_dir, args.train_batch_size)

    if args.use_repeat_sampler:
      print("use RepeatFactorTrainingSampler")
      train_sampler = RepeatFactorTrainingSampler(train_dataset.labels, repeat_thresh=0.5)
    else:
      train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    
    train_dataloader = torch.utils.data.DataLoader(
      train_dataset, sampler=train_sampler,
      batch_size=args.train_batch_size // get_world_size(),
      # collate_fn=utils.batch_list_to_batch_tensors
      collate_fn=basic_batch_convert
      )
    
    if args.data_name == 'hox_data':
      val_dataset = PriDatasetExt(args, args.data_dir_for_val, args.eval_batch_size)
    else:
      val_dataset = PriDataset(args, args.data_dir_for_val, args.eval_batch_size)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
      val_dataset, sampler=val_sampler,
      batch_size=args.eval_batch_size,
      # collate_fn=utils.batch_list_to_batch_tensors,
      collate_fn=basic_batch_convert,
      pin_memory=False)

    for i_epoch in range(int(start_epoch), int(start_epoch + args.num_train_epochs)):

      # learning_rate_decay(args, i_epoch, optimizer)
      if is_main_process():
        print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
        print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])

      train_sampler.set_epoch(i_epoch)
      
      train_one_epoch(model, train_dataloader, val_dataloader, optimizer, post_process, device, i_epoch, args)

      lr_scheduler.step()
      

if __name__ == '__main__':
  main()