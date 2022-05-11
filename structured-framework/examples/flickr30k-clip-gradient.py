import sys
import os

import json

sys.path.insert(1, os.getcwd())
from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_clip import Flickr30KClip
from transformers import CLIPTokenizer
from visualizations.visualizegradient import *

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

# get the model
analysismodel = Flickr30KClip(target_idx=target_idx)

# unimodal image gradient
"""
for instance_idx in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    instance = data.getdata(instance_idx)

    # get the model predictions
    preds = analysismodel.forward(instance)

    # compute and print grad saliency with and without multiply orig:
    saliency = get_saliency_map(instance, analysismodel, 0)
    grads = saliency[0]
    t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
    heatmap2d(
        t,
        f"visuals/flickr30k-clip-{instance_idx}-{target_idx}-saliency.png",
        instance[0],
    )
"""
instance_text_target_ids = {
    50: {"ids": [1, 2, 3], "text": "three small dogs"},
    100: {"ids": [15, 16, 17], "text": "his frying pan"},
    150: {"ids": [5, 6, 7, 8, 9], "text": "white facial and chest markings"},
    200: {"ids": [11, 12, 13, 14], "text": "white and orange tulips"},
    250: {"ids": [1, 2, 3, 4, 5], "text": "two boys, two girls"},
    300: {"ids": [6, 7, 8, 9, 10], "text": "black shirt and brown pants"},
    350: {"ids": [9], "text": "suitcase"},
    400: {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9],
        "text": "woman in jean jacket and black sunglasses",
    },
    450: {"ids": [2, 3, 4, 5, 6], "text": "white dog with brown ears"},
    500: {"ids": [7, 8, 9], "text": "pink food tray"},
}


for instance_idx in [350]:
    instance = data.getdata(instance_idx)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    grads, di, tids = analysismodel.getdoublegrad(
        instance, instance[-1], instance_text_target_ids[instance_idx]["ids"]
    )
    # print(tokenizer.convert_ids_to_tokens(tids[0].detach().cpu().numpy()))
    print(
        tokenizer.convert_ids_to_tokens(
            tids[0]
            .detach()
            .cpu()
            .numpy()[instance_text_target_ids[instance_idx]["ids"]]
        )
    )

    grads = grads[0]
    t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
    heatmap2d(
        t,
        f"visuals/flickr30k-clip-{instance_idx}-{target_idx}-doublegrad.png",
        instance[0],
    )
