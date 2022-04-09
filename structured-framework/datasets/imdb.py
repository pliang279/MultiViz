"""Implements dataloaders for IMDB dataset"""

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
    """Implements a torch Dataset class for the imdb dataset."""
    
    def __init__(self, file: h5py.File, split: str = None, vggfeature: bool = False) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.file = file
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
        self.vggfeature = vggfeature
        self.dataset = h5py.File(self.file, 'r')

    def getdata(self, ind):
        text = self.dataset["features"][ind + self.start_ind]
        image = self.dataset["images"][ind + self.start_ind] if not self.vggfeature else \
            self.dataset["vgg_features"][ind + self.start_ind]
        label = self.dataset["genres"][ind + self.start_ind]
        return text, image, label

    def length(self):
        return self.size

    def classnames(self):
        raise NotImplementedError

    def sample(self, num):
        sampled=[]
        nums = list(range(self.length()))
        random.shuffle(nums)
        idx = 0
        while(len(sampled) < num):
            a = self.getdata(nums[idx])
            sampled.append(a)
            idx += 1
        return sampled
