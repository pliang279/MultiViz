import torch
import sys
import os
sys.path.insert(1,os.getcwd())
from models.vqa_lxmert import VQALXMERT
from tqdm import tqdm
import random
from private_test_scripts.optimizer import BertAdam
from private_test_scripts.vqa_val3 import vallxmert
def train(analysismodel,points,epochs=1,ttotal = 40,bs = 32,processed = None):
    if processed is None:
        processed = []
        for point in tqdm(points):
            processed.append( (analysismodel.private_prep(point),point[-1]))
    model = analysismodel.model
    optimizer = torch.optim.SGD(model.answer_head.logit_fc[3].parameters(),lr=0.0001)
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
        torch.save(model,'debuggingrandom'+str(ep)+'.pt')
from private_test_scripts.vqa_val3 import vallxmert
if __name__=='__main__':
    for iz in range(10):
        m = VQALXMERT('cuda:0')
        train(m,torch.load('diff1280s'+str(iz)+'.pt'))
        #train(m,torch.load('alpoints.pt'))
        print(vallxmert(torch.load('debuggingrandom0.pt')))
        #print(vallxmert(torch.load('debuggingal0.pt')))
    








        




