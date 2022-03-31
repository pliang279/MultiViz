
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.emap import emap_print_report

# get the dataset
datas = VQADataset('val')
# get the model
analysismodel = VQALXMERT()
# pick data instance you want to explain
instance = datas.getdata(0)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# run emap report
emap_print_report(instance,datas.sample(99),'image','text',analysismodel,(predlabel,correctlabel))
