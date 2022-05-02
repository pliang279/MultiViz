from transformers import CLIPProcessor, CLIPModel
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


class Flickr30KClip(analysismodel):
    def __init__(self, device="cuda", target_idx=0):
        super(analysismodel, self).__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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

    # TODO: Check if this is needed for Flickr30k
    def getcorrectlabel(self, datainstance):
        pass

    def forward(self, datainstance):
        with torch.no_grad():
            im = PIL.Image.open(datainstance[0])
            sentences = [datainstance[1][self.target_idx]]
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
            return outputs.logits_per_image  # image-text similarity score

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self, datainstances):
        outs = []
        for di in datainstances:
            outs.append(self.forward(di))
        return outs

    def getlogitsize(self):
        return 1

    def getlogit(self, resultobj):
        return resultobj[0]

    # TODO: We cannot implement prelinear for this,
    # as there are two different heads
    def getprelinear(self, resultobj):
        pass

    # TODO: Check if this needs to be updated
    def getpredlabel(self, resultobj):
        return resultobj[0][[self.target_idx]].item()

    # TODO: We cannot implement prelinear for this,
    # as there are two different heads
    def getprelinearsize(self):
        pass

    def replaceunimodaldata(self, datainstance, modality, newinput):
        if modality == "image":
            randname = "tmp/flickr30k" + str(random.randint(0, 100000000)) + ".jpg"
            plt.imsave(randname, newinput)
            return randname, datainstance[1][self.target_idx]
        elif modality == "text":
            return datainstance[0], newinput
        else:
            raise ValueError

    def getgrad(self, datainstance, target_idx):
        im = PIL.Image.open(datainstance[0])
        sentences = [datainstance[1][self.target_idx]]
        inputs = self.processor(
            text=sentences,
            images=im,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        inputs.pixel_values.requires_grad = True
        outputs = self.model(**inputs)
        outputs.logits_per_image[0][target_idx].backward()
        return [inputs.pixel_values.detach()[0]], [inputs.pixel_values.grad.detach()[0]]

    # TODO: If this is correct
    def private_prep(self, datainstance):
        with torch.no_grad():
            im = PIL.Image.open(datainstance[0])
        sentences = [datainstance[1][self.target_idx]]
        inputs = self.processor(
            text=sentences,
            images=im,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return inputs
