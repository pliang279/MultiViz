
import os
import sys

sys.path.insert(1,os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.utils import loadvqalxmertfeats
from analysis.SparseLinearEncoding import get_sparse_linear_model
# get the dataset
datas = VQADataset('val')
# get the model
analysismodel = VQALXMERT(device='cuda:1')
# get saved features
trains=loadvqalxmertfeats(['tmp/VQAfeats/LXMERT_train_feats_1.pkl','tmp/VQAfeats/LXMERT_train_feats_2.pkl','tmp/VQAfeats/LXMERT_train_feats_3.pkl'])
vals=loadvqalxmertfeats(['tmp/VQAfeats/LXMERT_val_feats_1.pkl','tmp/VQAfeats/LXMERT_val_feats_2.pkl'])
# get the explanations
params,res = get_sparse_linear_model(analysismodel,trains,vals,vals)
# visualize sparsity-accuracy tradeoff
sparsityaccgraph(res,'visuals/vqaspartsityaccplot.png')



