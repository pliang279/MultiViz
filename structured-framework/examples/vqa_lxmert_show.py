import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
datas = VQADataset("val")
# get the model
analysismodel = VQALXMERT()


def makepic(id):
    instance = datas.getdata(id)
    predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
    datas.makepic(id, pr=predlabel)


for i in [
    6115,
    554,
    4272,
    6788,
    9383,
    3321,
    1162,
    5238,
    9702,
    8217,
    1031,
    8996,
    578,
    3979,
    3920,
]:
    makepic(i)
