from typing import Dict, List, Tuple, NamedTuple, Any
import math
import warnings
import bisect
from bisect import bisect_right
import torch
from torch.optim.lr_scheduler import _LRScheduler


### not ready
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        print([
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs 
        ])
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs 
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, 
      warmup_epochs=10, verbose=False):
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        # if self.last_epoch == 0:
        #     # return self.base_lrs
        # return [group['lr'] * self.gamma
        #         for group in self.optimizer.param_groups]
        ## self.last_epoch will from 0
        if self.last_epoch < self.warmup_epochs:
          # fac = (0.5 if self.last_epoch == 0 else self.last_epoch) / self.warmup_epochs
          fac = _get_warmup_factor(self.last_epoch, self.warmup_epochs)
          return [lr * fac for lr in self.base_lrs]

        elif self.last_epoch == self.warmup_epochs:
          return self.base_lrs

        else:
          return [group['lr'] * self.gamma
                  for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
          e = 0
        else:
          e = self.last_epoch - self.warmup_epochs
        return [base_lr * self.gamma ** e
                for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_epochs=10, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        last_epoch = self.last_epoch
        last_epoch2 = self.last_epoch - self.warmup_epochs
        if last_epoch < self.warmup_epochs:
          fac = _get_warmup_factor(self.last_epoch, self.warmup_epochs)
          return [lr * fac for lr in self.base_lrs]
        elif last_epoch == self.warmup_epochs:
          return self.base_lrs
        # if last_epoch == 0:
        #     return self.base_lrs
        elif (last_epoch2 - 1 - self.T_max) % (2 * self.T_max) == 0:
          return [group['lr'] + (base_lr - self.eta_min) *
                  (1 - math.cos(math.pi / self.T_max)) / 2
                  for base_lr, group in
                  zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * last_epoch2 / self.T_max)) /
                (1 + math.cos(math.pi * (last_epoch2 - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

def _get_warmup_factor(epoch: int, warmup_epochs: int) -> float:
  fac = (0.5 if epoch == 0 else epoch) / warmup_epochs
  return fac

def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        print(alpha, iter)
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))