import matplotlib.pyplot as plt
import torch
import json
import torch.nn.functional as F
import numpy as np
import copy
import cv2
from models.analysismodel import analysismodel
import models.mdetr.datasets.transforms as T
import PIL
import random


class CLEVRMDETR(analysismodel):
    def __init__(self,device = 'cuda'):
        super(analysismodel,self).__init__()
        self.device = device
        self.model = torch.load('mdetr/gqa_model.pt')
        self.dummy_info = None
        self.modalitynames = ['image','text']
        self.modalitytypes = ['image','text']

    # image transformations
    def make_coco_transforms(cautious=False):

        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        #max_size = 1333

        return T.Compose(
                [
                    #T.RandomResize([800], max_size=max_size),
                    normalize,
                ]
        )

    def get_normed(img, target):
        tmp = copy.deepcopy(target)
        img, _ = make_coco_transforms()(img, tmp)
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
            normed_image = get_normed(image, self.dummy_info)
            samples = torch.unsqueeze(normed_image, 0).to(self.device)
            captions = [datainstance[1]]

            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
            probas = torch.cat((outputs['pred_answer_obj'], outputs['pred_answer_attr'], 
                                outputs['pred_answer_rel'], outputs['pred_answer_global'], 
                                outputs['pred_answer_cat']), 1)

            return probas.cpu().detach().numpy() #,deepcopy(model_features)

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self,datainstances):
        outs=[]
        for di in datainstances:
            outs.append(self.forward(di))
        return outs
        
    def getlogitsize(self):
        return 0 #27

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




