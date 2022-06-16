
import sys
import os
sys.path.insert(1,os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from tqdm import tqdm
d=VQADataset()
m=VQALXMERT("cuda:1")

li = [ 143,  181,  941, 1139,  427]
from analysis.utils import *
feats = loadvqalxmertfeats(['tmp/VQAfeats/LXMERT_feats_2.pkl'])
ids=[]
for i in range(110000,160000):
    if d.getdata(i)[-1] is not None:
        ids.append(i)
pts=[]
zs = torch.zeros(len(ids),1536)
for i in range(len(ids)):
    zs[i] = feats[i][0]
for i in li:
    zss = zs[:,i]
    top10 = torch.argsort(torch.abs(zss))[-128:]
    total = 0
    totals = 0

    for j in tqdm(range(128)):
        id = ids[top10[j]]
        di = d.getdata(id)
        pts.append(di)
        res = m.forward(di)
        corr = m.getpredlabel(res)
        correct = m.getcorrectlabel(di)
        if corr != correct:
            logit = torch.nn.functional.softmax(m.getlogit(res))[corr]
            total += logit.item()
            totals += 1
    print(total/totals)
import torch
torch.save(pts,"debugpointsnew.pt")

