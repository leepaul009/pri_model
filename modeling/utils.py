
import torch
import torch.nn as nn

def c2_msra_fill(module: nn.Module) -> None:
  nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
  if module.bias is not None:
    nn.init.constant_(module.bias, 0)



