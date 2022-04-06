import os
import sys
sys.path.insert(1,'/home/paul/yiwei/multimodal_analysis/structured-framework')
sys.path.insert(1,'/home/paul/yiwei/multimodal_analysis/structured-framework/model/mdetr')


import matplotlib.pyplot as plt
import torch
import json
import torch.nn.functional as F
import numpy as np
import copy
import cv2
from model.analysismodel import analysismodel
import model.mdetr.datasets.transforms as T
#import torchvision.transforms as T
import PIL
import random

#from models.mdetr.models.mdetr import MDETR
from dataset.clevr import CLEVRDataset


class CLEVRMDETR(analysismodel):
    def __init__(self,device = 'cuda'):
        super(analysismodel,self).__init__()
        self.device = device
        self.model = torch.load('model/mdetr/clevr_model.pt')
        self.dummy_info = None
        self.modalitynames = ['image','text']
        self.modalitytypes = ['image','text']

    # image transformations
    def make_clevr_transforms(self, cautious=False):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #scales = [256, 288, 320, 352, 384]

        return T.Compose(
            [
                # T.RandomResize([480], max_size=1333),
                normalize,
            ]
        )

        raise ValueError(f"unknown {image_set}")

    def get_normed(self, img, target):
        tmp = copy.deepcopy(target)
        img, _ = self.make_clevr_transforms()(img, tmp)
        return img  
    # end image transformations


    def getunimodaldata(self,datainstance,modality):
        if modality == 'image':
            return np.asarray(PIL.Image.open(datainstance[0]))
        elif modality == 'text':
            return datainstance[1]
        else:
            raise ValueError

    def getcorrectlabel(self,datainstance):
        return datainstance[2]

    def forward(self,datainstance):
        with torch.no_grad():
            image = PIL.Image.open(datainstance[0]).convert('RGB')
            normed_image = self.get_normed(image, self.dummy_info)
            samples = torch.unsqueeze(normed_image, 0).to(self.device)
            captions = [datainstance[1]]

            memory_cache = self.model(samples, captions, encode_and_save=True)
            outputs = self.model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
            pred_answer_binary_comp = -outputs['pred_answer_binary']
            probas = torch.cat((outputs['pred_answer_binary'].unsqueeze(0).T, pred_answer_binary_comp.unsqueeze(0).T, 
                                outputs['pred_answer_attr'], outputs['pred_answer_reg']), 1)

            return probas.cpu().detach().numpy() #,deepcopy(model_features)

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self,datainstances):
        outs=[]
        for di in datainstances:
            outs.append(self.forward(di))
        return outs
        
    def getlogitsize(self):
        return 27

    def getlogit(self,resultobj):
        return resultobj[0]

    def getprelinear(self,resultobj):
        return None # resultobj[1]

    def getpredlabel(self,resultobj):
        return resultobj[0].argmax(-1)

    def getprelinearsize(self):
        return 0 #1536

    def replaceunimodaldata(self,datainstance,modality,newinput):
        if modality == 'image':
            randname = 'tmp/clevr'+str(random.randint(0,100000000))+'.jpg'
            plt.imsave(randname,newinput)
            return randname,datainstance[1],datainstance[2],datainstance[3]
        elif modality == 'text':
            return datainstance[0],newinput,datainstance[2],datainstance[3]
        else:
            raise ValueError


dataset = CLEVRDataset()
model = CLEVRMDETR()
res = model.forward(dataset.getdata(0))
print(model.getlogit(res))

