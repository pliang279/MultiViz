import os
import sys
import torch
import torchvision
from gensim.models import KeyedVectors
import h5py
import numpy as np
import random

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_lf import IMDb_LF
from analysis.gradientbased import get_saliency_map

random.seed(0)
np.random.seed(0)

table_path = '/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5'
dataset = h5py.File(table_path, 'r')
model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_best_lf_vgg_torch_yiwei.pth'
multibench_path = '/home/paul/nihalj/MultiBench'
word2vec_path = '/home/paul/nihalj/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz'
raw_imdb_root_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()

# get the dataset
train_dataset = IMDBDataset('train', raw_imdb_root_path, dataset=dataset, crop=False)
val_dataset = IMDBDataset('val', raw_imdb_root_path, dataset=dataset, crop=False)
test_dataset = IMDBDataset('test', raw_imdb_root_path, dataset=dataset, crop=False)

# get the model
analysismodel = IMDb_LF(model_path, multibench_path, word2vec, vgg16_model, device, batch_size=32)

# pick data instance you want to explain
instance = train_dataset.getdata(0)

# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)

# compute and print grad saliency with and without multiplyorig:
print(get_saliency_map(instance, analysismodel, predlabel))
print(get_saliency_map(instance, analysismodel, correctlabel))
print(get_saliency_map(instance, analysismodel, predlabel, multiplyorig=True))
print(get_saliency_map(instance, analysismodel, correctlabel, multiplyorig=True))