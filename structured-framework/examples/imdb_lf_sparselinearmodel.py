import os
import sys
import pandas as pd
import torch
import torchvision
from gensim.models import KeyedVectors

sys.path.insert(1,os.getcwd())

from datasets.imdb import IMDBDataset_pd
from models.imdb_lf import IMDb_LF
from models.imdb_raw_lf import IMDb_LF as IMDb_raw_LF
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *

df_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/datasets/mmimdb_extras/imdb_torch_vgg.pkl'
model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_best_lf_vgg_torch.pth'
multibench_path = '/home/paul/yiwei/MultiBench'

df = pd.read_pickle(df_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = IMDBDataset_pd(df, 'train')
val_dataset = IMDBDataset_pd(df, 'val')
test_dataset = IMDBDataset_pd(df, 'test')

analysismodel = IMDb_LF(model_path, multibench_path, device=device)

train_instances = train_dataset.sample(train_dataset.length())
val_instances = val_dataset.sample(val_dataset.length())
test_instances = test_dataset.sample(test_dataset.length())

# TODO: SLM currently only works for classification; Deal with multilabel classification SLM
# params,res = get_sparse_linear_model(analysismodel, getembeds(train_instances, analysismodel), 
#                                      getembeds(val_instances, analysismodel), 
#                                      getembeds(test_instances,analysismodel,
#                                      reallabel=True))

# sparsityaccgraph(res, 'visuals/imdb/imdb_spartsityaccplot.png', scatter=True)


params = torch.load('ckpt/sparselinearmodel.pt')

word2vec_path = '/home/paul/nihalj/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz'
raw_imdb_root_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()
analysis_model_raw = IMDb_raw_LF(model_path, multibench_path, word2vec, vgg16_model, device)

datainstance = train_dataset.getdata(5)
analyzefeaturesandvisualizeall(params, train_dataset.sample(1000), analysis_model_raw, 1, 
                               'visuals/imdb/imdb-lf-sparse-', prelinear=None, pathnum=30, k=5)
analyzepointandvisualizeall(params, datainstance, analysis_model_raw,
                            analysis_model_raw.getpredlabel(analysis_model_raw.forward(datainstance)),
                            'visuals/imdb/imdb-lf-sparse-new')