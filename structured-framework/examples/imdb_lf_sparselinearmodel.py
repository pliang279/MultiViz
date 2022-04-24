import os
import sys
import torch
import torchvision
from gensim.models import KeyedVectors
import h5py
import random

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_vgg_bert_lf import IMDb_LF
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *
from transformers import BertTokenizer, BertModel

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
# datas = IMDBDataset('val', raw_data_path, dataset=dataset)
train_dataset = IMDBDataset('train', raw_imdb_root_path, dataset=dataset, crop=False)
val_dataset = IMDBDataset('val', raw_imdb_root_path, dataset=dataset, crop=False)
test_dataset = IMDBDataset('test', raw_imdb_root_path, dataset=dataset, crop=False)
# get the model
analysismodel = IMDb_LF(model_path, multibench_path, bert_model, bert_tokenizer, vgg16_model, device, batch_size=32)


train_instances = train_dataset.sample(train_dataset.length())
val_instances = val_dataset.sample(val_dataset.length())
test_instances = test_dataset.sample(test_dataset.length())

# TODO: SLM currently only works for classification; Deal with multilabel classification SLM
params,res = get_sparse_linear_model(analysismodel, getembeds(train_instances, analysismodel), 
                                     getembeds(val_instances, analysismodel), 
                                     getembeds(test_instances,analysismodel,
                                     reallabel=True), modelsavedir='models/mmimdb_lf_extras/sparselinearmodel_vgg_bert.pt')

sparsityaccgraph(res, 'visuals/imdb/imdb_spartsityaccplot.png', scatter=True)


# params = torch.load('models/mmimdb_lf_extras/SLM_ckpt/sparselinearmodel.pt')

# datainstance = test_dataset.getdata(5)
# analyzefeaturesandvisualizeall(params, test_dataset.sample(1000), analysismodel, 1, 
#                                'visuals/imdb/imdb-lf-sparse-feature-', prelinear=None, pathnum=30, k=5)
# analyzepointandvisualizeall(params, datainstance, analysismodel,
#                             analysismodel.getpredlabel(analysismodel.forward(datainstance)),
#                             'visuals/imdb/imdb-lf-sparse-point-')