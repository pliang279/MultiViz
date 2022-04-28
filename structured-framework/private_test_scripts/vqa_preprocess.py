import torch
import os
import sys

sys.path.insert(1, os.getcwd())
from models.vqa_lxmert import VQALXMERT
from datasets.vqa import VQADataset
from tqdm import tqdm

datas = VQADataset("train")
analysismodel = VQALXMERT("cuda:1")
stores = []
for i in tqdm(range(datas.length())):
    # for i in tqdm(range(200)):
    datainstance = datas.getdata(i)
    correct = analysismodel.getcorrectlabel(datainstance)
    if correct is None:
        continue
    prep = analysismodel.private_prep(datainstance)
    p1, p2, p3 = prep
    stores.append([i, p1, p2, p3, correct])
    if i % 160000 == 159999:
        torch.save(stores, "preptrain" + str(i / 160000) + ".pt")
        stores = []

torch.save(stores, "preptrain2.pt")
