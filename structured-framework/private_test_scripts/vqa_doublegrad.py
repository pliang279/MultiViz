import sys
import os
sys.path.insert(1,os.getcwd())
from PIL import Image
from models.vqa_lxmert import VQALXMERT
import torch
from datasets.vqa import VQADataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def getrect(b,color):
    #print(b)

    bb = (b * 224).to(torch.int32).detach().cpu().numpy()
    rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], linewidth=1, edgecolor=color, facecolor='none')
    return rect

def producetop3(datainstance,analysismodel,targets,targetwords,savename,show=False):
    plt.clf()
    fig,ax=plt.subplots()
    image = analysismodel.getunimodaldata(datainstance,'image')
    ax.imshow(Image.fromarray(image))
    grad,bboxs=analysismodel.getdoublegrad(datainstance,targets,targetwords)
    #print(grad[0].size())
    gs = torch.abs(torch.sum(grad[0][0],dim=1))
    ids = torch.argsort(gs)
    bboxs = bboxs.cpu()[0]
    # first choice
    id = ids[-1]
    bb = bboxs[id]
    rect = getrect(bb,'r')
    ax.add_patch(rect)
    #"""
    # second choice
    id = ids[-2]
    bb = bboxs[id]
    rect = getrect(bb,'b')
    ax.add_patch(rect)
    # third choice
    id = ids[-3]
    bb = bboxs[id]
    rect = getrect(bb,'g')
    ax.add_patch(rect)
    #"""
    if show:
        plt.show()

    plt.savefig(savename)

def produceallfromsaved(datainstance,analysismodel):
    plt.clf()
    fig,ax=plt.subplots()
    image = analysismodel.getunimodaldata(datainstance,'image')
    ax.imshow(Image.fromarray(image))
    #print(grad[0].size())
    _ = analysismodel.forward(datainstance)
    bboxs = analysismodel.frcnnout['normalized_boxes'].cpu()[0]
    for bb in bboxs:
        #print('bb')
        rect = getrect(bb,'g')
        ax.add_patch(rect)
    plt.show()


#import numpy as np
#m = VQALXMERT("cuda:1")
#d = VQADataset()
#producetop3(d.getdata(54),m,np.arange(3129),[8],'hat.png')

