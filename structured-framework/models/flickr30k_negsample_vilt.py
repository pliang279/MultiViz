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


class Flickr30kNegsampleVilt(analysismodel):
    def __init__(self, device="cuda"):
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

    def getunimodaldata(self, datainstance, modality):
        if modality == "image":
            return cv2.resize(np.asarray(PIL.Image.open(datainstance[0])), (224, 224))
        elif modality == "text":
            return datainstance[1]
        else:
            raise ValueError

    def getcorrectlabel(self, datainstance):
        return datainstance[-1]

    def forward(self, datainstance):
        with torch.no_grad():
            im = PIL.Image.open(datainstance[0])
            sentences = datainstance[1]
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
            return (
                outputs.logits[0],
                vilt_pooler_output[0],
            )  # image-text similarity score

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self, datainstances):
        outs = []
        for di in datainstances:
            outs.append(self.forward(di))
        return outs

    def getlogitsize(self):
        return 2

        # We only get one logit, but for sparse linear model, we will return two

    def getlogit(self, resultobj):
        # We need to do this to get logical class values
        pos_softmax = F.softmax(resultobj[0])  # dimension 1
        neg_softmax = 1 - pos_softmax
        return torch.stack([pos_softmax, neg_softmax], dim=1)

    def getprelinear(self, resultobj):
        return resultobj[1]

    def getpredlabel(self, resultobj):
        if F.softmax(resultobj[0][0]).item() > 0.5:
            return 1
        else:
            return 0

    def getprelinearsize(self):
        return 768

    def replaceunimodaldata(self, datainstance, modality, newinput):
        if modality == "image":
            randname = "tmp/flickr30k" + str(random.randint(0, 100000000)) + ".jpg"
            plt.imsave(randname, newinput)
            return randname, datainstance[1]
        elif modality == "text":
            return datainstance[0], newinput
        else:
            raise ValueError

    def getgrad(self, datainstance, target):
        im = PIL.Image.open(datainstance[0])
        sentence = datainstance[1]
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
