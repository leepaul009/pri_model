from typing import Dict, List, Tuple, NamedTuple, Any

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import utils



# def cross_entropy(input, target, *, reduction="mean", **kwargs):
#     """
#     Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
#     for empty inputs.
#     """
#     if target.numel() == 0 and reduction == "mean":
#         return input.sum() * 0.0  # connect the gradient
#     return F.cross_entropy(input, target, reduction=reduction, **kwargs)

class BinLoss(nn.Module):
    def __init__(self, config):
        super(BinLoss, self).__init__()
        self.config = config
        # self.logic_loss = nn.CrossEntropyLoss(reduction="mean")
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    # # offset * bin_hwidth + bin_ctr = label_dg
    # loss_output, outputs = self.loss(
    #     h1, gt_bins, h2, gt_offsets, bin_ctrs, bin_half_widths)
    def forward(self, 
                logics : Tensor, # (bs, n_bins)
                gt_logics : Tensor, # (bs,)
                reg,  # (bs, 1)
                gt_reg # (bs, 1)
                ):
        
        # h1 = (bs, n_bins); gt_bins = (bs,)
        # loss_cls = self.logic_loss(logics, gt_logics)
        # pred_inds = torch.argmax(logics, dim=-1, keepdim=True) # (bs, 1)

        gt_logics = gt_logics.reshape(-1) # (bs,)
        loss_cls = F.cross_entropy(logics, gt_logics, reduction="sum")
        
        loss_reg = self.reg_loss(reg.reshape(-1, 1), 
                                 gt_reg.reshape(-1, 1))
        
        output = dict()
        num_reg = logics.shape[0]

        output['num_reg'] = num_reg
        output['cls_loss'] = loss_cls
        output['reg_loss'] = loss_reg
        output['loss'] = (loss_cls + loss_reg) / (num_reg + 1e-10)
        
        return output

class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        # self.loss = nn.MSELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor):
        """
        pred=(N, 1)
        target=(N, 1)
        """
        output = dict()
        num_reg = pred.shape[0]
        loss = self.reg_loss(pred, target)
        output['num_reg'] = num_reg
        output['reg_loss'] = loss
        output['loss'] = loss / (num_reg + 1e-10)
        return output
