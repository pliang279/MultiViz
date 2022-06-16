
import sys
import os
sys.path.insert(1,os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
from tqdm import tqdm
d=VQADataset()
m=VQALXMERT()

li=[871,24]
from analysis.utils import *
feats = loadvqalxmertfeats(['tmp/VQAfeats/LXMERT_val_feats_2.pkl'])
ids=[]
for i in range(110000,160000):
    if d.getdata(i)[-1] is not None:
        ids.append(i)

model = m.model.answer_head.logit_fc[3]
tocat = []
with torch.no_grad():
    for i in tqdm(range(len(ids))):
        tocat.append(model(feats[i][0].unsqueeze(0).to(m.device)))

catted = torch.cat(tocat, dim=0)
print(catted.size())


sftmx=torch.nn.functional.softmax(catted,dim=1)
entropy = torch.sum(sftmx * torch.log(sftmx),dim=1)

c=torch.argsort(entropy)
print(entropy[c[0]])
print(entropy[c[100]])
pts=[]
for i in c[0:640]:
    pts.append(d.getdata(ids[i]))

torch.save(pts,'alpoints.py')


