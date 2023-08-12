from typing import Dict, List, Tuple, NamedTuple, Any

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import utils

from modeling.layer import KMersNet


from scipy.stats import linregress
from sklearn import metrics as sklearn_metrics




class PostProcess(nn.Module):
    def __init__(self, output_dir):
        super(PostProcess, self).__init__()
        self.output_dir = output_dir

    def forward(self, out):
        '''
        outputs:
            post_out: used in append method
        '''
        post_out = dict()
        post_out['num_reg'] = out['num_reg']
        post_out['reg_loss'] = out['reg_loss']
        if 'cls_loss' in out:
            post_out['cls_loss'] = out['cls_loss']
        return post_out

    def append(self, 
               metrics: Dict, 
               post_out=None, 
               preds=None, 
               input=None) -> Dict:
        
        if len(metrics.keys()) == 0:
            for key in post_out:
                if key != "loss":
                    metrics[key] = 0.0
        
        # num_reg, re_loss
        for key in post_out:
            if key == 'reg_loss' or key == 'cls_loss':
                # use item() to get scalar value, 
                # otherwise the value belong to graph
                metrics[key] += post_out[key].item()
            else:
                metrics[key] += post_out[key]
            # print("post process: {} = {}".format(key, metrics[key]))

        # gather prediction and labels during validation and inference
        if preds is not None and input is not None:
            preds = preds.detach().cpu().numpy().reshape(-1) # (bs, 1) => (bs)
            if isinstance(input[0], Dict):
                labels = np.array(utils.get_from_mapping(input, 'label'))
            elif len(input) == 3: # for new model
                list_dict, _, _ = input
                labels = np.array([idict['label'] for idict in list_dict])
            else: # useless
                labels = np.array([idict['label'] for idict, _ in input])
            if "preds" not in metrics:
                metrics["preds"] = preds
                metrics["gts"] = labels
            else:
                metrics["preds"] = np.concatenate((metrics["preds"], preds))
                metrics["gts"] = np.concatenate((metrics["gts"], labels))

        return metrics
    
    # def set_output_dir(self, path):
    #     self.output_dir = path

    def display(self, metrics, epoch, step=None, lr=None, time=None, training=False):

        if 'reg_loss' not in metrics:
            print("reg loss not found in metrics, {}".format(metrics.keys()))
        if 'cls_loss' in metrics:
            loss        = (metrics["reg_loss"] + metrics["cls_loss"]) / (metrics["num_reg"] + 1e-10)
            loss_logic  = (metrics["cls_loss"]) / (metrics["num_reg"] + 1e-10)
            loss_reg    = (metrics["reg_loss"]) / (metrics["num_reg"] + 1e-10)
        else:
            loss        = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
            loss_logic  = 0.0
            loss_reg    = loss
        
        metrics["loss"] = loss
        
        # print info
        if training:
            print("epoch = {} step = {}, loss = {:.4f}, loss_logic = {:.4f}, loss_reg = {:.4f}, time = {:.2f}, lr = {:.5f}".format(
                   epoch, step, loss, loss_logic, loss_reg, time, lr))
        else:
            accuracy = 0.0
            rvalue, pvalue, rrmse = 0, 0, 0
            if "preds" in metrics and "gts" in metrics:
                preds = metrics["preds"]
                gts = metrics["gts"]
                slope,intercept,rvalue,pvalue,stderr = linregress(gts, preds)
                # rvalue 表示 皮尔森系数，越接近1越好，一般要到0.75以上，预测合格，pvalue表示检验的p值，需要小于0.05，严格一点需要小于0.01
                rrmse = sklearn_metrics.mean_squared_error(gts, preds)
                # rrmse 表示实验值和预测值之间的均方根误差，值越接近于0越好
                # accuracy
                toleration = 1.0
                abs_error = np.abs(preds - gts)
                indices = np.where(abs_error <= toleration)[0]
                accuracy = len(indices) * 1.0 / len(abs_error)

                metrics['val_info'] = {
                    epoch : {'loss': loss, 'rvalue': rvalue, 'pvalue': pvalue, 'rrmse': rrmse},
                }

            print("validation epoch {}: loss = {:.4f}, acc = {:.4f}, loss_logic = {:.4f}, loss_reg = {:.4f}, rvalue = {:.4f}, pvalue = {:.4f}, rrmse = {:.4f}".format(
                epoch, loss, accuracy, loss_logic, loss_reg, rvalue, pvalue, rrmse))

    # only run for validation and test
    def updateOutput(self, epoch, loss_val, metrics, gmetrics):
        gmetrics['val_info'].update(metrics['val_info'])
        # replace output only when this epoch's loss is lower
        if loss_val < gmetrics['min_eval_loss']:
            gmetrics['output'] = {
                'epoch': epoch, # int
                'preds': metrics["preds"], # np.array
                'labels': metrics["gts"],  # np.array
            }
            
        save_dir = os.path.join(self.output_dir, "prediction")
        if not os.path.exists(save_dir):
            print("Directory {} doesn't exist, create a new.".format(save_dir))
            os.makedirs(save_dir)

        output_file = os.path.join(save_dir, "validation_metrics")
        np.savez(output_file, metrics=gmetrics)
        
