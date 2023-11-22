import itertools
import math
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import pickle
import random
import zlib
from utilities import comm

class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """

    def __init__(self, dataset_dicts, repeat_thresh, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors = self._get_repeat_factors(dataset_dicts, repeat_thresh)
        self._int_part = torch.trunc(rep_factors) # get int part
        self._frac_part = rep_factors - self._int_part # get float part

        self._local_indices = self._infinite_indices()
        print("[repeated sampler] re-balanced dataset length (_local_indices.length) =  {}".format(len(self._local_indices)))

    def _get_bin_id(self, intervals, label):
      # other_ind = len(intervals[:-1])
    #   found = False
      for ind, pair in enumerate(zip(intervals[:-1], intervals[1:])):
        i, j = pair # max, min
        if label <= i and label > j:
        #   found = True
          return ind
    #   if not found:
    #     return other_ind
      print("[repeated sampler error] can not find label {} in interval {}".format(label, intervals))
      return 0

    def _get_repeat_factors(self, dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.
        Args:
            dataset_dicts: labels
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        # for dataset_dict in dataset_dicts:  # For each image (without repeats)
        #     cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        #     for cat_id in cat_ids:
        #         category_freq[cat_id] += 1
        
        
        # intervals = np.arange(0, -30, -2).astype(np.float32)
        # intervals = np.array([np.inf, -4., -6., -8., -10., -12., -14., -16., -np.inf], dtype=np.float32)
        intervals = np.array([np.inf, -6., -8., -10., -12., -14., -np.inf], dtype=np.float32)

        bin_id_by_data = []
        for label in dataset_dicts:
        #   instance = pickle.loads(zlib.decompress(data_dict))
          bin_id = self._get_bin_id(intervals, label)
          bin_id_by_data.append(bin_id)
          category_freq[bin_id] += 1

        num_items = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_items

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }
        # do not repeat last label
        category_rep[len(intervals)-1] = 1

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        # for dataset_dict in dataset_dicts:
        #     cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        #     rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
        #     rep_factors.append(rep_factor)

        for bin_id in bin_id_by_data:
          rep_factor = category_rep[bin_id]
          rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    # def __iter__(self):
    #     start = self._rank
    #     yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def __iter__(self):
      start = self._rank
      yield from itertools.islice(self._local_indices, start, None, self._world_size)
      # yield from self._local_indices

    def __len__(self):
      return int(len(self._local_indices) / self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        # while True:
        # Sample indices with repeats determined by stochastic rounding; each
        # "epoch" may have a slightly different size due to the rounding.
        indices = self._get_epoch_indices(g)
        if self._shuffle:
            randperm = torch.randperm(len(indices), generator=g)
            return indices[randperm]
        else:
            return indices

    def set_epoch(self, epoch):
      self._local_indices = self._infinite_indices()
    # def _infinite_indices(self):
    #     g = torch.Generator()
    #     g.manual_seed(self._seed)
    #     while True:
    #         # Sample indices with repeats determined by stochastic rounding; each
    #         # "epoch" may have a slightly different size due to the rounding.
    #         indices = self._get_epoch_indices(g)
    #         if self._shuffle:
    #             randperm = torch.randperm(len(indices), generator=g)
    #             yield from indices[randperm]
    #         else:
    #             yield from indices

