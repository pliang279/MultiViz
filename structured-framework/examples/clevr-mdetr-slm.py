import sys
import os

sys.path.insert(1, os.getcwd())
from tqdm import tqdm
from models.clevr_mdetr import CLEVRMDETR
from datasets.clevr import CLEVRDataset
from visualizations.visualizegradient import *
from visualizations.visualizesparselinearmodel import *

traindatas = CLEVRDataset("train")
datas = CLEVRDataset("val")
analysismodel = CLEVRMDETR()

# unimodal image gradient
"""
for i in tqdm(range(0,200)):
    instance=datas.getdata(i)
    correct=analysismodel.getcorrectlabel(instance)
    raw,grad,fi = analysismodel.getgrad(instance,correct)
    t=normalize255(torch.sum(torch.abs(grad),dim=0))
    heatmap2d(t,'visuals/gs/gs'+str(i)+'.png',fi)
"""
instance = datas.getdata(223)
# params,res = getresonly(torch.load('ckpt/',analysismodel,vals))
sampledata = [datas.getdata(i) for i in range(15000)]
sampletraindata = [datas.getdata(i) for i in range(15000)]
trains = [analysismodel.getprelinear(res).cpu() for res in analysismodel.forwardbatch(sampletraindata)]
vals = [analysismodel.getprelinear(res).cpu() for res in analysismodel.forwardbatch(sampledata)]
# params = torch.load("ckpt/clevrsparselinearmodel_random.pt")
params, res = get_sparse_linear_model(analysismodel,trains,vals,vals)
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
analyzepointandvisualizeallgrad(
    params,
    instance,
    analysismodel,
    predlabel,
    "visuals/tmp2/clevr-mdetr-sparse-223-",
    "visuals/alls/clevr-mdetr-sparse-223-",
    pathnum=95,
    k=5,
)
analyzefeaturesandvisualizeallgrad(
    params,
    sampledata,
    analysismodel,
    predlabel,
    "visuals/tmp2/clevr-mdetr-sparse-223-sampled-",
    "visuals/alls/clevr-mdetr-sparse-223-sampled-",
    prelinear=torch.load("plclevr.pt"),
    pathnum=95,
    k=5,
    pointsperfeat=3,
)
