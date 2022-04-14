import os
import sys
sys.path.insert(1,'/home/paul/yiwei/multimodal_analysis/structured-framework')
sys.path.insert(1,'/home/paul/yiwei/multimodal_analysis/structured-framework/models/mdetr')


import matplotlib.pyplot as plt
import torch
import json
import torch.nn.functional as F
import numpy as np
import copy
import cv2
from models.analysismodel import analysismodel
import models.mdetr.dataset.transforms as T
import PIL
import random

from datasets.clevr import CLEVRDataset
from analysis.unimodallime import*
from visualizations.visualizelime import visualizelime


class CLEVRMDETR(analysismodel):
    def __init__(self,device = 'cuda'):
        super(analysismodel,self).__init__()
        self.device = device
        self.model = torch.load('models/mdetr/clevr_model_2.pt')
        self.dummy_info = None
        self.modalitynames = ['image','text']
        self.modalitytypes = ['image','text']


    # non-api functions
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

    def pred_answer(self, outputs):
        idx = 0
        ans_type = int(outputs["pred_answer_type"].argmax(-1))
        if ans_type == 0:
            idx = int(not (outputs["pred_answer_binary"].sigmoid() > 0.5))
        elif ans_type == 1:
            idx = int(outputs["pred_answer_attr"].argmax(-1)) + 2
        else:
            idx = int(outputs["pred_answer_reg"].argmax(-1)) + 17
        return idx    
    # end non-api functions


    def getunimodaldata(self,datainstance,modality):
        if modality == 'image':
            return np.asarray(PIL.Image.open(datainstance[0]).convert('RGB'))
        elif modality == 'text':
            return datainstance[1]
        else:
            raise ValueError

    def getcorrectlabel(self,datainstance):
        return datainstance[3]

    def forward(self,datainstance):
        with torch.no_grad():
            image = PIL.Image.open(datainstance[0]).convert('RGB')
            normed_image = self.get_normed(image, self.dummy_info)
            samples = torch.unsqueeze(normed_image, 0).to(self.device)
            captions = [datainstance[1]]

            model_features=[]
            def hook(module, input, output):
                model_feat = input
                model_features.append(model_feat[0][0])
            handle1 = self.model.answer_type_head.register_forward_hook(hook)
            handle2 = self.model.answer_binary_head.register_forward_hook(hook)
            handle3 = self.model.answer_attr_head.register_forward_hook(hook)
            handle4 = self.model.answer_reg_head.register_forward_hook(hook)

            memory_cache = self.model(samples, captions, encode_and_save=True)
            outputs = self.model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
            pred_answer_binary_comp = -outputs['pred_answer_binary']
            probas = torch.cat((outputs['pred_answer_binary'].unsqueeze(0).T, pred_answer_binary_comp.unsqueeze(0).T, 
                                outputs['pred_answer_attr'], outputs['pred_answer_reg']), 1)

            handle1.remove()
            handle2.remove()        
            handle3.remove()        
            handle4.remove()                            

            return probas[0], outputs, torch.cat(model_features) #copy.deepcopy(outputs['pre_linear'])

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
        return resultobj[2]

    def getpredlabel(self,resultobj):
        return self.pred_answer(resultobj[1])

    def getprelinearsize(self):
        return 256 * 4

    def replaceunimodaldata(self,datainstance,modality,newinput):
        if modality == 'image':
            randname = 'tmp/clevr'+str(random.randint(0,100000000))+'.jpg'
            plt.imsave(randname,newinput)
            return randname,datainstance[1],datainstance[2],datainstance[3]
        elif modality == 'text':
            return datainstance[0],newinput,datainstance[2],datainstance[3]
        else:
            raise ValueError

    def getgrad(self, datainstance, target):
        self.model.zero_grad()
        imgfile = datainstance[0]
        image = PIL.Image.open(imgfile).convert('RGB')
        normed_image = self.get_normed(image, self.dummy_info).to(self.device)
        
        normed_image.requires_grad = True

        samples = torch.unsqueeze(normed_image, 0).to(self.device)
        captions = [datainstance[1]]

        memory_cache = self.model(samples, captions, encode_and_save=True)
        outputs = self.model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
        pred_answer_binary_comp = -outputs['pred_answer_binary']
        probas = torch.cat((outputs['pred_answer_binary'].unsqueeze(0).T, pred_answer_binary_comp.unsqueeze(0).T, 
                            outputs['pred_answer_attr'], outputs['pred_answer_reg']), 1)
        
    
        probas[0][target].backward()
        grad = normed_image.grad.detach()
        return normed_image, grad, imgfile

if __name__ == '__main__':
    dataset = CLEVRDataset()
    model = CLEVRMDETR()

    datainstance = dataset.getdata(5)
    '''print(datainstance)
    resobj = model.forward(datainstance)
    pred_label_idx = model.getpredlabel(resobj)
    classnames = dataset.classnames()

    print(datainstance, classnames[pred_label_idx])

    modalityname = 'image'
    modalitytype = 'image'
    labels = (pred_label_idx,)

    exp = rununimodallime(datainstance,modalityname,modalitytype,model,labels)
    #visualizelime(exp, 'image', pred_label_idx)

    pre_linear = model.getprelinear(resobj)
    print(pre_linear.shape)'''

    print(model.getgrad(datainstance, 0))
