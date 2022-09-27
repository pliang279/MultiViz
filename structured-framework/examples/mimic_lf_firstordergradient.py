import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.mimic import MIMICDataset
from models.mimic_lf import MIMICLF
from analysis.gradientbased import get_saliency_map
from visualizations.visualizegradient import heatmapts,textmap

# get the dataset
datas = MIMICDataset("test")
# get the model
analysismodel = MIMICLF("/home/anon/MultiBench/mimiclfbest.pt", "/home/anon/MultiBench")
# pick data instance you want to explain
instance = datas.getdata(10)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)


print(predlabel)
print(correctlabel)
import numpy as np
_,grads = analysismodel.getgrad(instance,predlabel)
textmap(datas.statics,grads[0],'staticgrad.png')
heatmapts(np.arange(24).tolist(),datas.timeseries,grads[1].transpose(0,1).cpu().numpy(),'tsgrad.png')
