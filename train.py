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

from modeling.graphnet import GraphNet
from modeling.post_process import PostProcess

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 


def lr_decay_by_steps(steps_update_lr, all_steps, step, optimizer):
  # cur_lr = None
  # for p in optimizer.param_groups:
  #   cur_lr = p['lr']
  #   break
  # delta_lr = (args.learning_rate - min_lr) / (all_steps / steps_update_lr)
  # print(( args.learning_rate - min_lr), (all_steps / steps_update_lr), delta_lr )
  if step > 1 and step % steps_update_lr == 0:
    for p in optimizer.param_groups:
      p['lr'] *= 0.993
    print("step {}, updated lr = {}, update lr every {} steps"
      .format(step, p['lr'], steps_update_lr))


def learning_rate_decay(args, i_epoch, optimizer, optimizer_2=None):
  epoch_update_lr = 100
  if i_epoch > 0 and i_epoch % epoch_update_lr == 0:
    for p in optimizer.param_groups:
      p['lr'] *= 0.75


def save_ckpt(model, opt, save_dir, epoch, step, overwrite=False, cp_name=None):
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

  if not overwrite:
    output_model_file = os.path.join(
      save_dir, "model.{}.{}.ckpt".format(epoch + 1, step))
  elif cp_name is None:
    output_model_file = os.path.join(save_dir, "model.ckpt")
  else:
    output_model_file = os.path.join(save_dir, "{}.ckpt".format(cp_name))
  
  print("save checkpoint to {}".format(output_model_file))
  torch.save(
    {
      "epoch": epoch, 
      "step": step,
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


def val(model, dataloader, post_process, epoch, device, args, gmetrics):
  model.eval()

  iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
  metrics = dict()
  
  for step, batch in enumerate(iter_bar):
    with torch.no_grad():
      loss, pred, loss_output = model(batch, device)
      post_out = post_process(loss_output)
      post_process.append(metrics, post_out, pred, batch)
  
  post_process.display(metrics, epoch)
  model.train()

  # save model with min validation loss
  loss_val = metrics['loss']
  if loss_val < gmetrics['min_eval_loss']:
    gmetrics['save'] = True
  # updateOutput should be run after display
  post_process.updateOutput(epoch, loss_val, metrics, gmetrics)
  gmetrics['min_eval_loss'] = min(loss_val, gmetrics['min_eval_loss'])


def train_one_epoch(model, train_dataloader, val_dataloader, 
                    optimizer,
                    post_process, device, i_epoch, args, gmetrics):
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
      post_process.display(metrics, 
        i_epoch, step, curr_lr, end_time - start_time, training=True)
      
    # if is_main_process and step > 5000 and step % save_iters == 0:
    #   save_ckpt(model, optimizer, save_dir, i_epoch, step, overwrite=True)
    
    if args.step_lr:
      lr_decay_by_steps(args.steps_update_lr, steps_sz, step, optimizer)

  # do eval after an epoch training
  if args.do_eval:
    val(model, val_dataloader, post_process, i_epoch, device, args, gmetrics)

  if is_main_process and gmetrics['save']:
    save_ckpt(model, optimizer, save_dir, i_epoch, step, overwrite=True)


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
  args.output_dir = save_dir

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

  ######
  config = dict()
  model = GraphNet(config, args)
  model = model.to(device)
  post_process = PostProcess(args.output_dir)
    
  if distributed:
    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True,
        device_ids=[args.local_rank], 
        output_device=args.local_rank)

  ######
  optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate,
    betas=(0.9, 0.98), weight_decay=0.03)

  start_epoch = 0
  if args.resume:
    ckpt_path = args.resume_path
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt["state_dict"])
    start_epoch = ckpt["epoch"] + 1
    optimizer.load_state_dict(ckpt["opt_state"])
    print("load ckpt from {} and train from epoch {}".format(ckpt_path, start_epoch))

  ######
  # use 20 epoch from init_lr to 0
  lr_scheduler = WarmupCosineAnnealingLR(
    optimizer, 
    T_max=args.num_train_epochs - args.warmup_epoch, 
    warmup_epochs=args.warmup_epoch)
  # lr_scheduler = WarmupExponentialLR(
  #   optimizer, gamma=0.95, warmup_epochs=args.warmup_epoch)
  
  
  if args.do_test:
    if args.data_name == 'hox_data':
      test_dataset = PriDatasetExt(args, args.data_dir_for_test, args.test_batch_size)
    else:
      test_dataset = PriDataset(args, args.data_dir_for_test, args.test_batch_size)

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

    if args.use_repeat_sampler:
      print("use RepeatFactorTrainingSampler")
      train_sampler = RepeatFactorTrainingSampler(train_dataset.labels, repeat_thresh=0.5)
    else:
      train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    
    train_dataloader = torch.utils.data.DataLoader(
      train_dataset, sampler=train_sampler,
      batch_size=args.train_batch_size // get_world_size(),
      collate_fn=utils.batch_list_to_batch_tensors)
    
    if args.data_name == 'hox_data':
      val_dataset = PriDatasetExt(args, args.data_dir_for_val, args.eval_batch_size)
    else:
      val_dataset = PriDataset(args, args.data_dir_for_val, args.eval_batch_size)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
      val_dataset, sampler=val_sampler,
      batch_size=args.eval_batch_size,
      collate_fn=utils.batch_list_to_batch_tensors,
      pin_memory=False)

    gmetrics = {
      'save': False, 
      'min_eval_loss': 1e6,
      'val_info': dict(),
      'output': None,
    }

    for i_epoch in range(int(start_epoch), int(start_epoch + args.num_train_epochs)):

      # learning_rate_decay(args, i_epoch, optimizer)
      if is_main_process():
        print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
        print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])

      train_sampler.set_epoch(i_epoch)
      
      train_one_epoch(model, train_dataloader, val_dataloader, 
                      optimizer, post_process, device, i_epoch, args, gmetrics)

      lr_scheduler.step()
      

if __name__ == '__main__':
  main()