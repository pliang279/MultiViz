
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.mimic import MIMICDataset
from models.mimic_lf import MIMICLF
from analysis.gradientbased import get_saliency_map

# get the dataset
datas = MIMICDataset('test')
# get the model
analysismodel = MIMICLF('/home/paul/MultiBench/mimiclfbest.pt','/home/paul/MultiBench')
# pick data instance you want to explain
instance = datas.getdata(0)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# compute and print grad saliency with and without multiplyorig:
print(get_saliency_map(instance,analysismodel,predlabel))
print(get_saliency_map(instance,analysismodel,correctlabel))
print(get_saliency_map(instance,analysismodel,predlabel,multiplyorig=True))
print(get_saliency_map(instance,analysismodel,correctlabel,multiplyorig=True))
