import torch.nn as nn
import copy

def get_clones(module: nn.Module, size: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(size)])
