import os
import sys
import torch
import torchvision
import h5py
import random

sys.path.insert(1, os.getcwd())

from datasets.flickr30k_negsample import Flickr30kNegsampleDataset
from models.flickr30k_negsample_vilt import Flickr30kNegsampleVilt
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *

random.seed(0)
np.random.seed(0)

os.makedirs("visuals/flickr30k_negsample", exist_ok=True)
os.makedirs("models/flickr30k_negsample_vilt", exist_ok=True)

# get the datasets
train_dataset = Flickr30kNegsampleDataset("train")
valid_dataset = Flickr30kNegsampleDataset("valid")
test_dataset = Flickr30kNegsampleDataset("test")


# get the model
analysismodel = Flickr30kNegsampleVilt()


train_instances = train_dataset.sample(train_dataset.length())
val_instances = valid_dataset.sample(valid_dataset.length())
test_instances = test_dataset.sample(test_dataset.length())

# train_instances = train_dataset.sample(5)
# val_instances = valid_dataset.sample(5)
# test_instances = test_dataset.sample(5)

# train_embeds = getembeds(train_instances, analysismodel)
# val_embeds = getembeds(val_instances, analysismodel)
# test_embeds = getembeds(test_instances, analysismodel)
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, instances, model):
        self.instances = instances
        self.model = model

    def __getitem__(self, index):

        return getembeds([self.instances[index]], self.model)[0]

    def __len__(self):
        return len(self.instances)


train_embeds = EmbeddingDataset(train_instances, analysismodel)
val_embeds = EmbeddingDataset(val_instances, analysismodel)
test_embeds = EmbeddingDataset(test_instances, analysismodel)


params, res = get_sparse_linear_model(
    analysismodel,
    train_embeds,
    val_embeds,
    test_embeds,
    modelsavedir="models/flickr30k_negsample_vilt/sparselinearmodel_vilt.pt",
)

sparsityaccgraph(
    res, "visuals/flickr30k_negsample/vilt_spartsityaccplot.png", scatter=True
)
