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


class IMDBDataset(Dataset):
    """Implements a torch Dataset class for the imdb dataset."""
    
    def __init__(self, file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')
        text = self.dataset["features"][ind+self.start_ind]
        image = self.dataset["images"][ind+self.start_ind] if not self.vggfeature else \
            self.dataset["vgg_features"][ind+self.start_ind]
        label = self.dataset["genres"][ind+self.start_ind]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size



def get_dataloader(path: str, num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40, vgg: bool = False):
    """Get dataloaders for IMDB dataset.

    Args:
        path (str): Path to training datafile.
        num_workers (int, optional): Number of workers to load data in. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        batch_size (int, optional): Batch size of data. Defaults to 40.
        vgg (bool, optional): Whether to return raw images or pre-processed vgg features. Defaults to False.
        skip_process (bool, optional): Whether to pre-process data or not. Defaults to False.
        no_robust (bool, optional): Whether to not use robustness measures as augmentation. Defaults to False.

    Returns:
        Tuple[Dict]: Tuple of Training dataloader, Validation dataloader, Test Dataloader
    """
    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg),
                                  shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)

    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg),
                                shuffle=False, num_workers=num_workers, batch_size=batch_size)

    test_dataloader = DataLoader(IMDBDataset(path, 18160, 25959, vgg),
                                     shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader



def main():
    path = "/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5"
    num_workers = 4
    train_shuffle = True
    batch_size = 128
    vgg = True
    trainloader, valloader, testloader = get_dataloader(path, num_workers, train_shuffle, batch_size, vgg)
    saved_model_fn = 'best_lf.pth'

    sys.path.append('/home/paul/MultiBench/')
    from unimodals.common_models import Linear, MaxOut_MLP
    from fusions.common_fusions import Concat
    from training_structures.Supervised_Learning import train, test

    encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
    head = Linear(1024, 23).cuda()
    fusion = Concat().cuda()

    train(encoders, fusion, head, trainloader, valloader, 1000, early_stop=True, task="multilabel",
        save=saved_model_fn, optimtype=torch.optim.AdamW, lr=8e-3, weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())


if __name__ == '__main__':
    main()