
import sys
import os
sys.path.insert(1,os.getcwd())
from datasets.vqa import VQADataset
d = VQADataset()
def getpts(diff):
    pts =[]
    for i in range(2000):
        pt = d.getdata(110000+25*i+diff*2)
        if pt[-1] is not None:
            pts.append(pt)
        if len(pts) == 12800:
            return pts

import torch
for diff in range(0,10):
    torch.save(getpts(diff),'diff1280s'+str(diff)+'.pt')

