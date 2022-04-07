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
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataset = IMDBDataset(data_path, 'val', vggfeature=True)
valloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

analysismodel = IMDb_LF(pretrained_model_path, multibench_path)