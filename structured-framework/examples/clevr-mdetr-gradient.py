import sys
import os
sys.path.insert(1,os.getcwd())
from tqdm import tqdm
from models.clevr_mdetr import CLEVRMDETR
from datasets.clevr import CLEVRDataset
from visualizations.visualizegradient import *
datas=CLEVRDataset('val')
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


