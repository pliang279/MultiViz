import os
import sys

sys.path.insert(1, os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.utils import loadvqalxmertfeats
from analysis.SparseLinearEncoding import get_sparse_linear_model, getresonly
from visualizations.visualizesparselinearmodel import *
import numpy as np

# get the dataset
datas = VQADataset("val")
# get the model
analysismodel = VQALXMERT(device="cuda:1")
# get saved features
# trains=loadvqalxmertfeats(['tmp/VQAfeats/LXMERT_train_feats_1.pkl','tmp/VQAfeats/LXMERT_train_feats_2.pkl','tmp/VQAfeats/LXMERT_train_feats_3.pkl'])
vals = loadvqalxmertfeats(
    ["tmp/VQAfeats/LXMERT_val_feats_1.pkl", "tmp/VQAfeats/LXMERT_val_feats_2.pkl"]
)
# get the explanations
# params,res = get_sparse_linear_model(analysismodel,trains,vals,vals)
params, res = getresonly(
    torch.load("ckpt/vqasparselinearmodel.pt"), analysismodel, vals
)
# visualize sparsity-accuracy tradeoff
sparsityaccgraph(res, "visuals/vqaspartsityaccplot.png", scatter=True)
# visualize lime results on features
sampledata = datas.getseqdata(0, 20000)
import torch

# pl = [analysismodel.getprelinear(res).cpu() for res in analysismodel.forwardbatch(sampledata)]
# torch.save(pl,'tmp/pl.pt')
pll = torch.load("tmp/pl.pt")
pl = torch.zeros(len(pll), len(pll[0]))
for i in range(len(pl)):
    pl[i] = pll[i]

# """
def analyze(i):
    datainstance = datas.getdata(i)
    predlabel = analysismodel.getpredlabel(analysismodel.forward(datainstance))
    # analyzefeaturesandvisualizeall(params, sampledata, analysismodel, predlabel, 'visuals/tmp/vqa-lxmert-sparse-'+str(i)+'-sampled-', 'visuals/alls/vqa-lxmert-sparse-'+str(i)+'-sampled-',  prelinear=pl.float(), pathnum=95, k=5,pointsperfeat=3)
    analyzepointandvisualizeall(
        params,
        datainstance,
        analysismodel,
        analysismodel.getpredlabel(analysismodel.forward(datainstance)),
        "visuals/tmp/vqa-lxmert-sparse-" + str(i) + "-",
        "visuals/alls/vqa-lxmert-sparse-" + str(i) + "-",
        pathnum=95,
        k=5,
    )


# for i in range(0,150):
#    analyze(i*50+5)
analyze(554)
# """
