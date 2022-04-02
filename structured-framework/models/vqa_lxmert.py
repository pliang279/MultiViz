from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import matplotlib.pyplot as plt
import torch
import json
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import cv2
from models.analysismodel import analysismodel
from models.lxmert_extras.utils import Config
from models.lxmert_extras.processing_image import Preprocess
from models.lxmert_extras.modeling_frcnn import GeneralizedRCNN
import PIL
import random


class VQALXMERT(analysismodel):
    def __init__(self,device = 'cuda'):
        super(analysismodel,self).__init__()
        self.device=device
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(self.device)
        self.image_preprocess = Preprocess(self.frcnn_cfg) 
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(self.device)
        self.model = self.lxmert_vqa
        self.modalitynames = ['image','text']
        self.modalitytypes = ['image','text']
    def getunimodaldata(self,datainstance,modality):
        if modality == 'image':
            return cv2.resize(np.asarray(PIL.Image.open(datainstance[0])),(224,224))
        elif modality == 'text':
            return datainstance[1]
        else:
            raise ValueError
    def getcorrectlabel(self,datainstance):
        return datainstance[-1]
    def forward(self,datainstance):
        with torch.no_grad():
            images,sizes,scales_yx = self.image_preprocess(datainstance[0])
            output_dict = self.frcnn(
                images.to(self.device),
                sizes.to(self.device),
                scales_yx=scales_yx.to(self.device),
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
                location=self.device
            )
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")
            inputs = self.lxmert_tokenizer(
                [datainstance[1]],
                padding="max_length",
                max_length=20,
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            model_features=None
            def hook(module, input, output):
                nonlocal model_features
                model_feat = input[0][0].squeeze()
                model_features = model_feat
            handle = self.lxmert_vqa.answer_head.logit_fc[-1].register_forward_hook(hook)
            output_vqa = output_vqa = self.lxmert_vqa(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                visual_feats=features.to(self.device),
                visual_pos=normalized_boxes.to(self.device),
                token_type_ids=inputs.token_type_ids.to(self.device),
                output_attentions=False,
            )
            handle.remove()
            assert model_features.shape[0] == 1536
            return output_vqa['question_answering_score'][0],deepcopy(model_features)
    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self,datainstances):
        outs=[]
        for di in datainstances:
            outs.append(self.forward(di))
        return outs
    def getlogitsize(self):
        return 3129
    def getlogit(self,resultobj):
        return resultobj[0]
    def getprelinear(self,resultobj):
        return resultobj[1]
    def getpredlabel(self,resultobj):
        return resultobj[0].argmax(-1).item()
    def getprelinearsize(self):
        return 1536
    def replaceunimodaldata(self,datainstance,modality,newinput):
        if modality == 'image':
            randname = 'tmp/vqa'+str(random.randint(0,100000000))+'.jpg'
            plt.imsave(randname,newinput)
            return randname,datainstance[1],datainstance[2],datainstance[3]
        elif modality == 'text':
            return datainstance[0],newinput,datainstance[2],datainstance[3]
        else:
            raise ValueError




