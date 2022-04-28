import os
import sys
import torch

multibench_path = "/home/paul/yiwei/MultiBench"
sys.path.append(multibench_path)

from torch.utils.data import Dataset, DataLoader
import h5py
import os
import sys
from typing import *


class IMDBDataset(Dataset):
    def __init__(
        self, file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False
    ) -> None:
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind - start_ind
        self.vggfeature = vggfeature

    def __getitem__(self, ind):
        if not hasattr(self, "dataset"):
            self.dataset = h5py.File(self.file, "r")
        text = self.dataset["features"][ind + self.start_ind]
        image = (
            self.dataset["images"][ind + self.start_ind]
            if not self.vggfeature
            else self.dataset["vgg_features"][ind + self.start_ind]
        )
        label = self.dataset["genres"][ind + self.start_ind]

        return text, image, label

    def __len__(self):
        return self.size


def get_dataloader(
    path: str,
    test_path: str,
    num_workers: int = 8,
    train_shuffle: bool = True,
    batch_size: int = 40,
    vgg: bool = False,
    skip_process=False,
    no_robust=False,
) -> Tuple[Dict]:
    train_dataloader = DataLoader(
        IMDBDataset(path, 0, 15552, vgg),
        shuffle=train_shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    val_dataloader = DataLoader(
        IMDBDataset(path, 15552, 18160, vgg),
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    if no_robust:
        test_dataloader = DataLoader(
            IMDBDataset(path, 18160, 25959, vgg),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        return train_dataloader, val_dataloader, test_dataloader


from unimodals.common_models import Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train

data_path = "/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5"

traindata, validdata, testdata = get_dataloader(
    data_path, "../video/mmimdb", vgg=True, batch_size=128, no_robust=True
)

encoders = [
    MaxOut_MLP(512, 512, 300, linear_layer=False).cuda(),
    MaxOut_MLP(512, 1024, 4096, 512, False).cuda(),
]
head = Linear(1024, 23).cuda()
fusion = Concat().cuda()


model_save_fn = "imdb_best_lf.pth"
train(
    encoders,
    fusion,
    head,
    traindata,
    validdata,
    1000,
    early_stop=True,
    task="multilabel",
    save=model_save_fn,
    optimtype=torch.optim.AdamW,
    lr=8e-3,
    weight_decay=0.01,
    objective=torch.nn.BCEWithLogitsLoss(),
)
