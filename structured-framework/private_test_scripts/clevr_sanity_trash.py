import sys
import os
sys.path.insert(1,os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
with open('data/CLEVR_v1.0/scenes/CLEVR_val_scenes_bbox.json') as f:
    jsons = json.load(f)['scenes']
def drawallbbox(instance):
    im=Image.open(instance[0])
    fig,ax = plt.subplots()
    ax.imshow(im)
    idx = int(instance[0][-10:-4])
    objs = jsons[idx]['objects']
    for obj in objs:
        bbox = obj['bbox'] 
        rect = patches.Rectangle((bbox[0],bbox[2]),bbox[1]-bbox[0],bbox[3]-bbox[2],linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig('qs.png')

def parse(sent):
    words = []
    for word in sent[:-1].split(" "):
        words.append(word)
    words.append("?")
    words.insert(0, "<start>")
    return words
from datasets.clevr import CLEVRDataset
from models.clevr_cnnlstmsa import CLEVRCNNLSTMSA
#from models.clevr_mdetr import parse
from visualizations.visualizegradient import *
m=CLEVRCNNLSTMSA('cpu')
d=CLEVRDataset()
import torch
import math
def computeboxgrad(grads,boxx,margin=10,avg=0):
    totalgrad = 0
    box = shrinkbox(boxx)
    for i in range(len(grads[0])):
        if i < box[0]-margin or i >= box[1] + margin:
            continue
        for j in range(len(grads)):
            if j < box[2]-margin or j >= box[3] + margin:
                continue
            totalgrad += grads[j][i].item()
    if avg == 1:
       return totalgrad/float((box[1]-box[0]+2*margin)*(box[3]-box[2]+2*margin))
    elif avg == 2: 
       return totalgrad/math.sqrt(float((box[1]-box[0]+2*margin)*(box[3]-box[2]+2*margin)))
    else:
       return totalgrad

def drawtriple(instance,targetwords,margin=10):
    drawspecificbbox(instance,targetwords,avg=0) 
    drawspecificbbox(instance,targetwords,avg=1)
    drawspecificbbox(instance,targetwords,avg=2,showgrad=True)
def shrinkbox(bbox):
    return bbox[0]*224 // 480, bbox[1]*224 // 480, bbox[2]*224//320,bbox[3]*224//320
def drawspecificbbox(instance,targetwords,show=True,margin=10,avg=0,showgrad=False):
    im=Image.open(instance[0])
    fig,ax = plt.subplots()
    ax.imshow(im)
    idx = int(instance[0][-10:-4])
    objs = jsons[idx]['objects']
    m.model.zero_grad()
    grad,_,txtids = m.getdoublegrad(instance,0,targetwords,True)
    grads = torch.sum(torch.abs(grad).squeeze(),dim=0)
    print(im.size)
    zc = parse(instance[1])
    for i in range(len(zc)):
        print(str(i)+" " + str(txtids[0][i].item())+" "+zc[i])
    impobj=None
    secondimpobj=None
    largestgrad=0.0
    secondlargestgrad = 0.0
    for obj in objs:
        bbox = obj['bbox'] 
        gr = computeboxgrad(grads,bbox,margin=margin,avg=avg)
        if gr > secondlargestgrad:
            if gr > largestgrad:
                secondlargestgrad = largestgrad
                largestgrad = gr
                secondimpobj=impobj
                impobj=obj
            else:
                secondlargestgrad = gr
                secondimpobj=obj
    bbox=impobj['bbox']
    rect = patches.Rectangle((bbox[0],bbox[2]),bbox[1]-bbox[0],bbox[3]-bbox[2],linewidth=1, edgecolor='r', facecolor='none')
    bbox = secondimpobj['bbox']
    rect2 = patches.Rectangle((bbox[0],bbox[2]),bbox[1]-bbox[0],bbox[3]-bbox[2],linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.savefig('qqs.png')
    if show:
        plt.show()
    if showgrad:
        plt.clf()
        t=normalize255(grads)
        heatmap2d(t,'gss2.png',instance[0])
        plt.show()

            
#drawspecificbbox(d.getdata(10),[13,14,15,16])

