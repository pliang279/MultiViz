"""Implements dataloaders for IMDB dataset to return raw (image, text) data"""

from tqdm import tqdm
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import h5py
from gensim.models import KeyedVectors
import os
import sys
import numpy as np
import torch
import random


class IMDBDataset:
    
    def __init__(self, file: h5py.File, split: str = None) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): ...
            split(str): Type of split
        """
        
        if split == 'train':
            self.start_ind = 0
            self.end_ind = 15552
        elif split == 'val':
            self.start_ind = 15552
            self.end_ind = 18160
        else:
            self.start_ind = 18160
            self.end_ind = 25959

        self.size = self.end_ind - self.start_ind

        raise NotImplementedError
        
        self.file = ...
        self.dataset = ...

    def getdata(self, ind):
        raise NotImplementedError
        text = ...
        image = ...
        label = ...
        return text, image, label

    def length(self):
        return self.size

    def classnames(self):
        raise NotImplementedError

    def sample(self, num):
        raise NotImplementedError

        sampled=[]
        nums = list(range(self.length()))
        random.shuffle(nums)
        idx = 0
        while(len(sampled) < num):
            a = self.getdata(nums[idx])
            sampled.append(a)
            idx += 1
        return sampled
