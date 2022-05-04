from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import matplotlib.pyplot as plt
import torch
import json
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import cv2
from models.analysismodel import analysismodel
import PIL
import random


class Flickr30KVilt(analysismodel):
    def __init__(self, device="cuda", target_idx=0):
        super(analysismodel, self).__init__()
        self.device = device
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-flickr30k"
        )
        self.model = ViltForImageAndTextRetrieval.from_pretrained(
            "dandelin/vilt-b32-finetuned-flickr30k"
        ).to(self.device)
        self.modalitynames = ["image", "text"]
        self.modalitytypes = ["image", "text"]
        self.target_idx = target_idx

    def getunimodaldata(self, datainstance, modality):
        if modality == "image":
            return cv2.resize(np.asarray(PIL.Image.open(datainstance[0])), (224, 224))
        elif modality == "text":
            return datainstance[1][self.target_idx]
        else:
            raise ValueError

    # TODO: Check how to do this for Flickr30K
    def getcorrectlabel(self, datainstance):
        pass

    def forward(self, datainstance):
        with torch.no_grad():
            im = PIL.Image.open(datainstance[0])
            sentences = datainstance[1][self.target_idx]
            inputs = self.processor(
                text=sentences,
                images=im,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            outputs = self.model(**inputs)

            vilt_pooler_output = self.model.vilt(**inputs).pooler_output
            return outputs.logits[0], vilt_pooler_output[0] # image-text similarity score

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self, datainstances):
        outs = []
        for di in datainstances:
            outs.append(self.forward(di))
        return outs

    def getlogitsize(self):
        return 1 # We only get one logit, need to use binary cross entropy during training

    def getlogit(self, resultobj):
        return resultobj[0]

    def getprelinear(self, resultobj):
        return resultobj[1]

    def getpredlabel(self, resultobj):
        return resultobj[0][[self.target_idx]].item()

    def getprelinearsize(self):
        return 768

    def replaceunimodaldata(self, datainstance, modality, newinput):
        if modality == "image":
            randname = "tmp/flickr30k" + str(random.randint(0, 100000000)) + ".jpg"
            plt.imsave(randname, newinput)
            return randname, datainstance[1][self.target_idx]
        elif modality == "text":
            return datainstance[0], newinput
        else:
            raise ValueError

    def getgrad(self, datainstance, target):
        im = PIL.Image.open(datainstance[0])
        sentence = datainstance[1][self.target_idx]
        inputs = self.processor(
            text=sentence,
            images=im,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        inputs.pixel_values.requires_grad = True
        outputs = self.model(**inputs)
        outputs.logits[0].backward()
        return [inputs.pixel_values.detach()[0]], [inputs.pixel_values.grad.detach()[0]]

    # TODO: If this is correct
    def private_prep(self, datainstance):
        with torch.no_grad():
            im = PIL.Image.open(datainstance[0])
            sentence = datainstance[1][self.target_idx]
            inputs = self.processor(
                text=sentence,
                images=im,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
        return inputs
