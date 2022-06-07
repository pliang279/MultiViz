import sys
import os

import json

sys.path.insert(1, os.getcwd())
from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_vilt import Flickr30KVilt
from transformers import ViltProcessor
import torch.nn.functional as F
from visualizations.visualizegradient import *
import random
from analysis.gradientbased import get_saliency_map

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

# get the model
analysismodel = Flickr30KVilt(target_idx=target_idx)

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
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-saliency.png",
        instance[0],
    )
"""
instance_text_target_ids = {
    50: {"ids": [1, 2, 3, 4, 5, 6], "text": "two black and white homeless men"},
    100: {"ids": [1, 2], "text": "the car"},
    150: {"ids": [1, 2], "text": "two dogs"},
    200: {"ids": [13, 14, 15, 16], "text": "shallow wading pool"},
    250: {"ids": [2, 3, 4], "text": "soccer team player"},
    300: {"ids": [1, 2, 3, 4, 5], "text": "two boys, two girls"},
    350: {"ids": [2, 3], "text": "little girl"},
    400: {"ids": [8, 9], "text": "black necklace"},
    450: {
        "ids": [4, 5, 6],
        "text": "the red shirt",
    },
    500: {"ids": [15, 16], "text": "the foothills"},
}

logits_and_props = {}
random.seed(42)
for instance_idx in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    instance = data.getdata(instance_idx)
    # Select another instance randomly which isn't same as current
    target_idx = random.choice([idx for idx in range(data.length())])
    while target_idx == instance_idx:
        target_idx = random.choice([idx for idx in range(data.length())])

    negative_instance = data.getdata(target_idx)

    instance = list(instance)
    instance[-1] = negative_instance[-1]

    with open(f"visuals/flickr30k-vilt-{instance_idx}-{0}-text.txt", "w") as f:
        f.write(instance[-1][0])

    # compute and print grad saliency with and without multiply orig:
    saliency = get_saliency_map(instance, analysismodel, 0)
    grads = saliency[0]

    t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
    heatmap2d(
        t,
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-saliency.png",
        instance[0],
    )

    probs, _ = analysismodel.forward(instance)

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-flickr30k")
    grads, di, tids = analysismodel.getdoublegrad(
        instance, instance[-1], instance_text_target_ids[instance_idx]["ids"]
    )

    print(
        dict(
            enumerate(
                processor.tokenizer.convert_ids_to_tokens(
                    tids[0].detach().cpu().numpy()
                )
            )
        )
    )

    print(
        processor.tokenizer.convert_ids_to_tokens(
            tids[0]
            .detach()
            .cpu()
            .numpy()[instance_text_target_ids[instance_idx]["ids"]]
        )
    )

    logits = probs.detach().cpu().numpy()[0]
    probs = F.softmax(probs).detach().cpu().numpy()[0]

    print(logits, probs)
    logits_and_props[instance_idx] = {"logits": logits, "probs": probs}

    grads = grads[0]
    t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
    heatmap2d(
        t,
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-doublegrad.png",
        instance[0],
    )

print(logits_and_props)
