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
from torch.utils.data import DataLoader

from utilities.comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict
import utils

from data.dataset_pri import PriDataset, PriDatasetExt
from data.distributed_sampler import RepeatFactorTrainingSampler
from utilities.schedular import WarmupExponentialLR, WarmupCosineAnnealingLR

from modeling.backbone.build_model import build_esm_dbert_model

from modeling.post_process import PostProcess


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
  gmetrics['save'] = False
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

    if is_main_process() and step % args.display_steps == 0:
      end_time = time.time()
      curr_lr = optimizer.param_groups[0]['lr']
      post_process.display(metrics, 
        i_epoch, step, curr_lr, end_time - start_time, training=True)
      
    # if is_main_process() and step > 5000 and step % save_iters == 0:
    #   save_ckpt(model, optimizer, save_dir, i_epoch, step)
    
    if args.step_lr:
      lr_decay_by_steps(args, steps_sz, step, optimizer)

  # do eval after an epoch training
  if args.do_eval:
    val(model, val_dataloader, post_process, i_epoch, device, args, gmetrics)

  if is_main_process() and gmetrics['save']:
    print("save model with best performance")
    save_ckpt(model, optimizer, save_dir, i_epoch, step, overwrite=True)
  
  if is_main_process() and args.save_model_epoch:
    print("save model per epoch")
    save_ckpt(model, optimizer, save_dir, i_epoch, step, overwrite=False)


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

  #########################################
  ### build model, batch converter, post processer and optimizer
  #########################################
  ### model
  model, basic_batch_convert = build_esm_dbert_model(args)
  model = model.to(device)
  
  post_process = PostProcess(args.output_dir)

  if distributed:
    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True,
        device_ids=[args.local_rank], 
        output_device=args.local_rank)

  ### optimizer
  optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate,
    betas=(0.9, 0.98), weight_decay=args.weight_decay)

  start_epoch = 0
  if args.resume:
    ckpt_path = args.resume_path
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt["state_dict"])
    start_epoch = ckpt["epoch"] + 1
    optimizer.load_state_dict(ckpt["opt_state"])
    print("load ckpt from {} and train from epoch {}".format(ckpt_path, start_epoch))

  ### learning rate schedular
  lr_scheduler = WarmupCosineAnnealingLR(
    optimizer, 
    eta_min = args.learning_rate * 0.01,
    T_max = args.num_train_epochs - args.warmup_epoch, # use 20 epoch from init_lr to 0
    warmup_epochs = args.warmup_epoch)
  # lr_scheduler = WarmupExponentialLR(
  #   optimizer, gamma=0.95, warmup_epochs=args.warmup_epoch)
  
  
  #########################################
  ### create data loader
  ######################################### 
  if not args.do_test:
    ### build training dataset
    train_dataset = PriDataset(args, args.data_dir, args.train_batch_size)

    ### build sampler
    if args.use_repeat_sampler:
      print("use RepeatFactorTrainingSampler")
      train_sampler = RepeatFactorTrainingSampler(train_dataset.labels, repeat_thresh=0.5)
    else:
      train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank())
    
    ### build dataloader
    train_dataloader = DataLoader(train_dataset, 
                                  sampler = train_sampler,
                                  batch_size = args.train_batch_size // get_world_size(),
                                  # collate_fn = utils.batch_list_to_batch_tensors
                                  collate_fn = basic_batch_convert)
    
    ### build training dataset
    val_dataset    = PriDataset(args, args.data_dir_for_val, args.eval_batch_size)
    val_sampler    = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, 
                                sampler = val_sampler,
                                batch_size = args.eval_batch_size,
                                # collate_fn = utils.batch_list_to_batch_tensors,
                                collate_fn = basic_batch_convert,
                                pin_memory = False)


    #########################################
    ### train and validation
    ######################################### 
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
  
  else:
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

if __name__ == '__main__':
  main()