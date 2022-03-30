
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.unimodallime import rununimodallime

datas = VQADataset('val')
analysismodel = VQALXMERT()
instance = datas.getdata(0)
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
correctlabel = analysismodel.getcorrectlabel(instance)
explanation = rununimodallime(instance,'image','image',analysismodel,(predlabel,correctlabel))
