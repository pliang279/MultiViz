import numpy as np

def tryconverttonp(num):
    try:
        return num.cpu().detach().numpy()
    except:
        #print(num)
        return num



def accuracy(preds, labels, k=1):
    correct = 0
    total = 0
    for i in range(preds.shape[0]):
        pred = preds[i, :k]
        label = labels[i]
        correct += (pred == label).any()
        total += 1
    return correct / total
import torch
import pickle
import pandas as pd
def loadpickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def processvqalxmertfeat(f,startid):
    a=loadpickle(f)
    li=[]
    for i in range(len(a)):
        row = a.iloc[i]
        embed = row.lxmert_features
        label = row.gt_answer_class
        li.append([embed,label,i+startid])
    return li

def processvqalxmertfeatsbinary(f):
    a=loadpickle(f)
    li=[]
    for i in range(len(a)):
        row = a.iloc[i]
        embed = row.lxmert_features
        label1 = row.gt_answer_class
        label2 = row.lxmert_answer_class
        if label1 == label2:
            li.append([torch.abs(embed),0])
        else:
            li.append([torch.abs(embed),1])
    return li

def loadvqalxmertfeats(fs):
    alls=[]
    for f in fs:
        alls += processvqalxmertfeat(f,len(alls))
    return alls


