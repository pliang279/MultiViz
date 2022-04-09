import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.imdb import IMDBDataset
from models.imdb_lf import IMDb_LF
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *

data_path = '/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5'
pretrained_model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/visuals/imdb_best_lf.pth'
multibench_path = '/home/paul/yiwei/MultiBench'

train_dataset = IMDBDataset(data_path, 'train', vggfeature=True)
val_dataset = IMDBDataset(data_path, 'val', vggfeature=True)
test_dataset = IMDBDataset(data_path, 'test', vggfeature=True)

analysismodel = IMDb_LF(pretrained_model_path, multibench_path)

train_instances = train_dataset.sample(train_dataset.length())
val_instances = val_dataset.sample(val_dataset.length())
test_instances = test_dataset.sample(test_dataset.length())

# TODO: SLM currently only works for classification; Deal with multilabel classification SLM
params,res = get_sparse_linear_model(analysismodel, getembeds(train_instances, analysismodel), 
                                     getembeds(val_instances, analysismodel), 
                                     getembeds(test_instances,analysismodel,
                                     reallabel=True))