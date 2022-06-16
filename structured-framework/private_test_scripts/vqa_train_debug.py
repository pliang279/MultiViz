import torch
import sys
import os
sys.path.insert(1,os.getcwd())
from models.vqa_lxmert import VQALXMERT
from tqdm import tqdm
import random
from private_test_scripts.optimizer import BertAdam
from private_test_scripts.vqa_val3 import vallxmert
def train(analysismodel,points,epochs=1,ttotal = 20,bs = 32,processed = None, lr=0.0001):
    if processed is None:
        processed = []
        for point in tqdm(points):
            processed.append( (analysismodel.private_prep(point),point[-1]))
    model = analysismodel.model
    optimizer = torch.optim.SGD(model.answer_head.logit_fc[3].parameters(),lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)
    device = analysismodel.device
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        random.shuffle(processed)
        for tt in tqdm(range(ttotal)):
            loss = 0.0
            model.zero_grad()
            for i in range(bs):
                ttbs, ans = processed[tt*bs+i]
                normalized_boxes,features,inputs = ttbs
                output = model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                    visual_feats=features.to(device),
                    visual_pos=normalized_boxes.to(device),
                    token_type_ids=inputs.token_type_ids.to(device),
                    output_attentions=False,
                )
                logits = output['question_answering_score']
                loss += criterion(logits,torch.LongTensor([ans]).to(device))
            loss.backward()
            optimizer.step()
        torch.save(model,'debugging'+str(ep)+'new.pt')

from datasets.vqa import VQADataset
d=VQADataset()
def getpts(a):
    selected = []
    for i in range(2500):
        dd = d.getdata(110000+i*20+a*3)
        if dd[-1] is not None:
            selected.append(dd)
        if len(selected)==640:
            break
    return selected


#from private_test_scripts.vqa_val3_debug import vallxmert 
if __name__=='__main__':
    pts = torch.load('debugpointsnew.pt')
    for lr in [0.0001,0.00005,0.00002,0.00001]:
        for iz in range(5):
            m = VQALXMERT('cuda:1')
            train(m,getpts(iz),lr=lr)
            print(vallxmert(m.model))
   








        




