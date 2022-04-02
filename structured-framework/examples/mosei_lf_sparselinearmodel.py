
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.mosei import MOSEIDataset
from models.mosei_lf import MOSEILF
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *
# get the dataset

datas1 = MOSEIDataset('train')
datas2 = MOSEIDataset('val')
datas3 = MOSEIDataset('test')
# get the model
analysismodel = MOSEILF('/home/paul/yiwei/MultiBench/mosei_lf_best.pt','/home/paul/yiwei/MultiBench')
# get train/valid/test data
instance1 = datas1.sample(datas1.length())
instance2 = datas2.sample(datas2.length())
instance3 = datas3.sample(datas3.length())
# get the explanations
params,res = get_sparse_linear_model(analysismodel,getembeds(instance1,analysismodel),getembeds(instance2,analysismodel),getembeds(instance3,analysismodel,reallabel=True))
# visualize sparsity-accuracy tradeoff
sparsityaccgraph(res,'visuals/moseispartsityaccplot.png')
# get data point to analyze
datainstance = instance3[0]
# visualize unimodal lime on top 5 features
analyzepointandvisualizeall(params,datainstance,analysismodel,analysismodel.getpredlabel(analysismodel.forward(datainstance)),'visuals/vqa-lxmert-sparse')
