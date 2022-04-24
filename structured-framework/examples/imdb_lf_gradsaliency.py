import os
import sys
import torch
import torchvision
from gensim.models import KeyedVectors
import h5py
import numpy as np
import random
from transformers import BertTokenizer, BertModel

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_vgg_bert_lf import IMDb_LF
from analysis.gradientbased import get_saliency_map

random.seed(0)
np.random.seed(0)

table_path = '/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5'
dataset = h5py.File(table_path, 'r')
model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_lf_vgg_bert.pth'
multibench_path = '/home/paul/nihalj/MultiBench'
raw_imdb_root_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# get the dataset
train_dataset = IMDBDataset('train', raw_imdb_root_path, dataset=dataset, crop=False)
val_dataset = IMDBDataset('val', raw_imdb_root_path, dataset=dataset, crop=False)
test_dataset = IMDBDataset('test', raw_imdb_root_path, dataset=dataset, crop=False)

# get the model
analysismodel = IMDb_LF(model_path, multibench_path, bert_model, bert_tokenizer, vgg16_model, device, batch_size=32)

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