
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.dime import dime
# get the dataset
datas = VQADataset('val')
# get the model
analysismodel = VQALXMERT()
# pick data instance you want to explain
instances = []
for i in range(100):
    instances.append( datas.getdata(i*50+5))
instance = instances[0]
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# run emap report
imageexp,textexp=dime(instances,0,analysismodel,(predlabel,correctlabel),samplematrix=torch.load('tmp/samplematrix.pt'))
