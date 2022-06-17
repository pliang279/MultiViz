import sys
import os
sys.path.insert(1,os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from visualizations.visualizesparselinearmodel import onefeature
import torch
d=VQADataset()
am = VQALXMERT('cuda:1')

sampledata = d.getseqdata(0, 20000)
pll = torch.load("tmp/pl.pt")
pl = torch.zeros(len(pll), len(pll[0]))
for i in range(len(pl)):
    pl[i] = pll[i]
for z in [1134]:
    onefeature(d,am,z,"visuals/tmp11/vqa","visuals/alls11/vqa",sampledata,pl,"visuals/data/vqa-val-")
