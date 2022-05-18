import sys
import os

import json

sys.path.insert(1, os.getcwd())
from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_vilt import Flickr30KVilt
from transformers import ViltProcessor
import torch.nn.functional as F
from visualizations.visualizegradient import *
from tqdm.auto import tqdm

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
# instance_text_target_ids = {
#     50: {"ids": [1, 2, 3], "text": "three small dogs"},
#     100: {"ids": [20, 21, 22], "text": "frying pan"},
#     150: {"ids": [5, 6, 7, 8, 9], "text": "white facial and chest markings"},
#     200: {"ids": [11, 12, 13, 14, 15, 16], "text": "white and orange tulips"},
#     250: {"ids": [1, 2, 3, 4, 5], "text": "two boys, two girls"},
#     300: {"ids": [6, 7, 8, 9, 10], "text": "black shirt and brown pants"},
#     350: {"ids": [9], "text": "suitcase"},
#     400: {
#         "ids": [2, 3, 4, 5, 6, 7, 8, 9],
#         "text": "woman in a jean jacket and black sunglasses",
#     },
#     450: {"ids": [2, 3, 4, 5, 6], "text": "white dog with brown ears"},
#     500: {"ids": [7, 8, 9], "text": "pink food tray"},
#     550: {"ids": [13, 14], "text": "fishing net"},
#     600: {"ids": [6, 7], "text": "orange scarf"},
#     650: {"ids": [7, 8, 9], "text": "a marble building"},
#     700: {"ids": [5, 6], "text": "black jacket"},
#     750: {"ids": [2, 3, 4, 5, 6, 7, 8], "text": "youmg man in white t-shirt"},
#     800: {"ids": [1, 2], "text": "five children"},
#     850: {"ids": [6, 7, 8, 9], "text": "pink knitted hat"},
#     900: {"ids": [1, 2, 3], "text": "a football player"},
#     950: {"ids": [1, 2, 3, 4, 5, 6, 7], "text": "two young girls wearing hijabs"},
#     1000: {"ids": [1, 2, 3, 4], "text": "a group of woman"},
# }

# instance_text_target_ids = {
#     50: {"ids": [3], "text": "dogs"},
#     100: {"ids": [22], "text": "pan"},
#     150: {"ids": [3], "text": "dog"},
#     200: {"ids": [14, 15, 16], "text": "tulips"},
#     250: {"ids": [2], "text": "boys"},
#     300: {"ids": [7], "text": "shirt"},
#     350: {"ids": [9], "text": "suitcase"},
#     400: {
#         "ids": [9],
#         "text": "sunglasses",
#     },
#     450: {"ids": [6], "text": "ears"},
#     500: {"ids": [9], "text": "tray"},
#     550: {"ids": [9], "text": "men"},
#     600: {"ids": [14], "text": "knife"},
#     650: {"ids": [7, 8, 9], "text": "a marble building"},
#     700: {"ids": [5, 6], "text": "black jacket"},
#     750: {"ids": [20], "text": "luggage"},
#     800: {"ids": [2], "text": "children"},
#     850: {"ids": [6, 7, 8, 9], "text": "pink knitted hat"},
#     900: {"ids": [6], "text": "football"},
#     950: {"ids": [3], "text": "girls"},
#     1000: {"ids": [4], "text": "woman"},
# }

# logits_and_props = {
#     50: {"logits": 9.413469, "probs": 1.0},
#     100: {"logits": 4.0988297, "probs": 1.0},
#     150: {"logits": 6.418624, "probs": 1.0},
#     200: {"logits": 9.75457, "probs": 1.0},
#     250: {"logits": 4.427314, "probs": 1.0},
#     300: {"logits": 8.706685, "probs": 1.0},
#     350: {"logits": 3.697362, "probs": 1.0},
#     400: {"logits": 8.945576, "probs": 1.0},
#     450: {"logits": 7.6222305, "probs": 1.0},
#     500: {"logits": 10.043411, "probs": 1.0},
#     550: {"logits": 3.5183542, "probs": 1.0},
#     600: {"logits": 6.1360574, "probs": 1.0},
#     650: {"logits": 8.271604, "probs": 1.0},
#     700: {"logits": 7.0360327, "probs": 1.0},
#     750: {"logits": 10.154418, "probs": 1.0},
#     800: {"logits": 9.210152, "probs": 1.0},
#     850: {"logits": 10.14699, "probs": 1.0},
#     900: {"logits": 5.792915, "probs": 1.0},
#     950: {"logits": 6.1690993, "probs": 1.0},
#     1000: {"logits": 8.121008, "probs": 1.0},
# }

# for instance_idx in [
#     50,
#     100,
#     150,
#     200,
#     250,
#     300,
#     350,
#     400,
#     450,
#     500,
#     550,
#     600,
#     650,
#     700,
#     750,
#     800,
#     850,
#     900,
#     950,
#     1000,
# ]:
#     instance = data.getdata(instance_idx)
#     # probs, _ = analysismodel.forward(instance)

#     processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-flickr30k")
#     grads, di, tids = analysismodel.getdoublegrad(
#         instance, instance[-1], instance_text_target_ids[instance_idx]["ids"]
#     )

#     print(dict(enumerate(processor.tokenizer.convert_ids_to_tokens(tids[0].detach().cpu().numpy()))))

#     # print(
#     #     processor.tokenizer.convert_ids_to_tokens(
#     #         tids[0]
#     #         .detach()
#     #         .cpu()
#     #         .numpy()[instance_text_target_ids[instance_idx]["ids"]]
#     #     )
#     # )

#     # logits = probs.detach().cpu().numpy()[0]
#     # probs = F.softmax(probs).detach().cpu().numpy()[0]

#     # logits_and_props[instance_idx] = {"logits": logits, "probs": probs}

#     grads = grads[0]
#     t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
#     heatmap2d(
#         t,
#         f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-doublegrad.png",
#         instance[0],
#     )

# print(logits_and_props)


logits_and_props = {}
#     50: {"logits": 9.413469, "probs": 1.0},
#     100: {"logits": 4.0988297, "probs": 1.0},
#     150: {"logits": 6.418624, "probs": 1.0},
#     200: {"logits": 9.75457, "probs": 1.0},
#     250: {"logits": 4.427314, "probs": 1.0},
#     300: {"logits": 8.706685, "probs": 1.0},
#     350: {"logits": 3.697362, "probs": 1.0},
#     400: {"logits": 8.945576, "probs": 1.0},
#     450: {"logits": 7.6222305, "probs": 1.0},
#     500: {"logits": 10.043411, "probs": 1.0},
#     550: {"logits": 3.5183542, "probs": 1.0},
#     600: {"logits": 6.1360574, "probs": 1.0},
#     650: {"logits": 8.271604, "probs": 1.0},
#     700: {"logits": 7.0360327, "probs": 1.0},
#     750: {"logits": 10.154418, "probs": 1.0},
#     800: {"logits": 9.210152, "probs": 1.0},
#     850: {"logits": 10.14699, "probs": 1.0},
#     900: {"logits": 5.792915, "probs": 1.0},
#     950: {"logits": 6.1690993, "probs": 1.0},
#     1000: {"logits": 8.121008, "probs": 1.0},
# }
print(data.length())
f = open(
    f"visuals/flickr30k-vilt-logits-probs.jsonl", "w"
)
for instance_idx in tqdm(range(0, data.length())):
    instance = data.getdata(instance_idx)
    for target_idx in [0, 1, 2, 3, 4]:
        analysismodel = Flickr30KVilt(target_idx=target_idx)

        probs, _ = analysismodel.forward(instance)

        # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-flickr30k")
        # grads, di, tids = analysismodel.getdoublegrad(
        #     instance, instance[-1], instance_text_target_ids[instance_idx]["ids"]
        # )

        # print(dict(enumerate(processor.tokenizer.convert_ids_to_tokens(tids[0].detach().cpu().numpy()))))

        # print(
        #     processor.tokenizer.convert_ids_to_tokens(
        #         tids[0]
        #         .detach()
        #         .cpu()
        #         .numpy()[instance_text_target_ids[instance_idx]["ids"]]
        #     )
        # )

        logits = probs.detach().cpu().numpy()[0]
        probs = F.softmax(probs).detach().cpu().numpy()[0]

        logits_and_props[instance_idx] = {"logits": logits, "probs": probs}

        # grads = grads[0]
        # t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
        # heatmap2d(
        #     t,
        #     f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-doublegrad.png",
        #     instance[0],
        # )

        # Write the logits and probabilities to a jsonl file

        f.write(f"{{\"logit\": {logits}, \"prob\": {probs}}}\n")

f.close()
# print(logits_and_props)