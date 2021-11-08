import random

import torch

import numpy as np


def set_seed(value=42):
    """Set random seed for everything.
    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)
