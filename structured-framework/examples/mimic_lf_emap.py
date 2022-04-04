
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.mimic import MIMICDataset
from models.mimic_lf import MIMICLF

from analysis.emap import emap_print_report

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
# run emap report
emap_print_report(instance,datas.sample(99),'static','timeseries',analysismodel,(predlabel,correctlabel))
