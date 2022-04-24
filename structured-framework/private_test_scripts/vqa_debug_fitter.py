import os
import sys
sys.path.insert(1,os.getcwd())

from analysis.utils import processvqalxmertfeatsbinary

vals=processvqalxmertfeatsbinary('tmp/VQAfeats/LXMERT_val_feats_1.pkl')

import torch

model=torch.nn.Linear(1536,1).cuda()

loader = torch.utils.data.DataLoader(vals,batch_size=100,shuffle=True)

optim = torch.optim.SGD(model.parameters(),lr=0.0001)

criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(300):
    totalloss = 0.0
    totals=0
    for j in loader:
        optim.zero_grad()
        out=model(j[0].cuda())
        loss = criterion(out.squeeze(),j[1].float().cuda())
        totalloss += loss.item()
        loss.backward()
        optim.step()
        totals += 1
    print("epoch "+str(epoch)+" loss "+str(totalloss/totals))

torch.save(model,'binaryerrorpred.pt')

    



