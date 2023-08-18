

from typing import Dict, List, Tuple
from copy import deepcopy
import numpy as np
import torch

from modeling.dbert.data import Alphabet as DAlphabet

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}


def mask_tokens(inputs: torch.Tensor, 
                tokenizer: DAlphabet, 
                is_extent: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    mlm_probability = 0.025
    
    mask_list = [-2, -1, 1, 2, 3] # MASK_LIST[tokenizer.kmer]


    labels = inputs.clone() # (B,T)
    
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability) # 0.025
    
    # get mask to spec tok (1 for spec tok, 0 for seq tok)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
    ] # BxList
    
    # fill zero for spec tok, and 0.025 for seq tok
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # set some of value of tensor as True (to mask it in training)
    masked_indices = torch.bernoulli(probability_matrix).bool() # (B,T)

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
      # get last index that is non-zero, which is index of last item in seq
      # print(inputs[i])
      
      tmp = torch.where(probability_matrix[i]!=0)[0]
      end = tmp.tolist()[-1]
      # if len(tmp) != 0:
      #   end = tmp.tolist()[-1]
      # else:
      #   end = 0
      
      # get indexes where mask is True
      mask_centers = set(torch.where(masked_index==1)[0].tolist())
      new_centers = deepcopy(mask_centers)
      if is_extent:
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
      new_centers = list(new_centers)
      masked_indices[i][new_centers] = True
    
    # non-mask item of sequence become -100
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_idx

    # 10% of the time, we replace masked input tokens with random word
    # chosen-mask item, but not be masked at previous step
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs #, labels