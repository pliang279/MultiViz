
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.dime import dime
from visualizations.visualizelime import visualizelime
import torch
# get the dataset
datas = VQADataset('val')
# get the model
analysismodel = VQALXMERT()
# pick data instance you want to explain
instance = datas.getdata(554)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# run dime
instances = []
for i in range(100):
    instances.append(datas.getdata(i*50+4))
samplematrix = torch.load('tmp/samplematrix.pt')
explanations = dime(instances,11,analysismodel,[predlabel,correctlabel],samplematrix)
# visualize explanations and save to directory
visualizelime(explanations[0],'image',0,'visuals/vqa-lxmert-554-image-dime-pred-uni.png')
visualizelime(explanations[0],'image',2,'visuals/vqa-lxmert-554-image-dime-correct-uni.png')
visualizelime(explanations[0],'image',1,'visuals/vqa-lxmert-554-image-dime-pred-multi.png')
visualizelime(explanations[0],'image',3,'visuals/vqa-lxmert-554-image-dime-correct-multi.png')
visualizelime(explanations[1],'text',0,'visuals/vqa-lxmert-554-text-dime-pred-uni.png')
visualizelime(explanations[1],'text',2,'visuals/vqa-lxmert-554-text-lime-correct-uni.png')
visualizelime(explanations[1],'text',1,'visuals/vqa-lxmert-554-text-dime-pred-multi.png')
visualizelime(explanations[1],'text',3,'visuals/vqa-lxmert-554-text-lime-correct-multi.png')

