"""
Class to handle Slow Influence Function on a PyTorch model.
Adapted from:
- https://github.com/allenai/allennlp/tree/main/allennlp/interpret/influence_interpreters
- https://github.com/xhan77/influence-function-analysis
"""

import torch
from typing import List
from torch.data.utils import Dataset, DataLoader
import re


class Influence:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.data.utils.Dataset,
        test_dataset: torch.data.utils.Dataset,
        params_to_freeze: List[str],
        use_cuda: bool = False,
    ):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.params_to_freeze = params_to_freeze

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        self.model.to(self.device)

        if params_to_freeze is not None:
            for name, param in self.model.named_parameters():
                if any([re.match(pattern, name) for pattern in params_to_freeze]):
                    param.requires_grad = False
