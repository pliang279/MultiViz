import torch
import os
import sys

sys.path.insert(1, os.getcwd())
from models.vqa_lxmert import VQALXMERT
from datasets.vqa import VQADataset
from tqdm import tqdm
datas = VQADataset('val')
analysismodel = VQALXMERT('cuda:1')
stores = []
for i in tqdm(range(160000,datas.length())):

    datainstance = datas.getdata(i)
    correct = analysismodel.getcorrectlabel(datainstance)
    if correct is None:
        continue
    prep = analysismodel.private_prep(datainstance)
    p1,p2,p3 = prep
    stores.append([i,p1,p2,p3,correct])
torch.save(stores,'prepval3new.pt')

    



