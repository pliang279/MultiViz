import os
import sys

sys.path.insert(1, os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT

import torch

datas = VQADataset("val")
analysismodel = VQALXMERT(device="cuda:1")
pll = torch.load("tmp/pl.pt")
pl = torch.zeros(len(pll), len(pll[0]))
for i in range(len(pl)):
    pl[i] = pll[i]
sampledata = datas.getseqdata(0, 20000)

from visualizations.visualizesparselinearmodel import *

topk = [871, 427, 70, 24, 941]
makefeats(
    topk,
    sampledata,
    analysismodel,
    "private_test_scripts/visuals/vqa-feats-",
    "private_test_scripts/visuals/vqa-feats-all-",
    pl,
    pointsperfeat=10,
)
