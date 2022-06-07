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
import copy

random.seed(42)
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
target_ids_100 = {
    0: "[CLS]",
    1: "a",
    2: "large",
    3: "bearded",
    4: "man",
    5: "flip",
    6: "##s",
    7: "a",
    8: "cr",
    9: "##ep",
    10: "##e",
    11: "or",
    12: "om",
    13: "##ele",
    14: "##t",
    15: "in",
    16: "mid",
    17: "##air",
    18: "with",
    19: "his",
    20: "fry",
    21: "##ing",
    22: "pan",
    23: ".",
    24: "[SEP]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_100 = {
    "100_1": {"ids": [2], "text": "large"},
    "100_2": {"ids": [3], "text": "bearded"},
    "100_3": {"ids": [4], "text": "man"},
    "100_4": {"ids": [8, 9, 10], "text": "crepe"},
    "100_5": {"ids": [12, 13, 14], "text": "omelet"},
    "100_6": {"ids": [20, 21, 22], "text": "frying pan"},
    "100_7": {"ids": [2, 3, 4], "text": "large bearded man"},
    "100_8": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "text": "large bearded man flips a crepe",
    },
    "100_9": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "text": "large bearded man flips a crepe or omelet",
    },
    "100_10": {"ids": [12, 13, 14, 15, 16, 17], "text": "omelet in mid air"},
}

target_ids_150 = {
    0: "[CLS]",
    1: "a",
    2: "black",
    3: "dog",
    4: "with",
    5: "white",
    6: "facial",
    7: "and",
    8: "chest",
    9: "markings",
    10: "standing",
    11: "in",
    12: "chest",
    13: "high",
    14: "water",
    15: ".",
    16: "[SEP]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_150 = {
    "150_1": {"ids": [2], "text": "black"},
    "150_2": {"ids": [3], "text": "dog"},
    "150_3": {"ids": [5], "text": "white"},
    "150_4": {"ids": [6], "text": "facial"},
    "150_5": {"ids": [8], "text": "chest"},
    "150_6": {"ids": [9], "text": "markings"},
    "150_7": {"ids": [10], "text": "standing"},
    "150_8": {"ids": [11], "text": "in"},
    "150_9": {"ids": [12], "text": "chest"},
    "150_10": {"ids": [13], "text": "high"},
    "150_11": {"ids": [14], "text": "water"},
    "150_12": {"ids": [2, 3], "text": "black dog"},
    "150_13": {"ids": [5, 6], "text": "white facial"},
    "150_14": {"ids": [5, 6, 7, 8, 9], "text": "white facial and chest markings"},
    "150_15": {"ids": [12, 13, 14], "text": "chest high water"},
}

target_ids_200 = {
    0: "[CLS]",
    1: "a",
    2: "man",
    3: "is",
    4: "taking",
    5: "photographs",
    6: "of",
    7: "a",
    8: "large",
    9: "garden",
    10: "of",
    11: "white",
    12: "and",
    13: "orange",
    14: "tu",
    15: "##lip",
    16: "##s",
    17: ".",
    18: "[SEP]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_200 = {
    "200_1": {"ids": [2], "text": "man"},
    "200_2": {"ids": [5], "text": "photographs"},
    "200_3": {"ids": [9], "text": "garden"},
    "200_4": {"ids": [11], "text": "white"},
    "200_5": {"ids": [13], "text": "orange"},
    "200_6": {"ids": [14, 15, 16], "text": "tulips"},
    "200_7": {"ids": [1, 2, 3, 4, 5], "text": "a man is taking photographs"},
    "200_8": {"ids": [8, 9], "text": "large garden"},
    "200_9": {
        "ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "text": "a man is taking photographs of a large garden",
    },
    "200_10": {
        "ids": [8, 9, 10, 11, 12, 13, 14, 15, 16],
        "text": "a large garden of white and orange tulips",
    },
    "200_11": {"ids": [11, 12, 13, 14, 15, 16], "text": "white and orange tulips"},
    "200_12": {
        "ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "text": "a man is taking photographs of a large garden of white and orange tulips",
    },
}

target_ids_500 = {
    0: "[CLS]",
    1: "a",
    2: "little",
    3: "girl",
    4: "in",
    5: "front",
    6: "a",
    7: "pink",
    8: "food",
    9: "tray",
    10: "is",
    11: "getting",
    12: "her",
    13: "bike",
    14: "helmet",
    15: "on",
    16: "by",
    17: "a",
    18: "woman",
    19: ".",
    20: "[SEP]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_500 = {
    "500_1": {"ids": [2], "text": "little"},
    "500_2": {"ids": [3], "text": "girl"},
    "500_3": {"ids": [7], "text": "pink"},
    "500_4": {"ids": [8], "text": "food"},
    "500_5": {"ids": [9], "text": "tray"},
    "500_6": {"ids": [13], "text": "bike"},
    "500_7": {"ids": [14], "text": "helmet"},
    "500_8": {"ids": [18], "text": "woman"},
    "500_9": {"ids": [2, 3], "text": "little girl"},
    "500_10": {"ids": [8, 9], "text": "food tray"},
    "500_11": {"ids": [7, 8, 9], "text": "pink food tray"},
    "500_12": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9],
        "text": "little girl in front a pink food tray",
    },
    "500_13": {"ids": [13, 14], "text": "bike helmet"},
}

target_ids_50 = {
    0: "[CLS]",
    1: "three",
    2: "small",
    3: "dogs",
    4: ",",
    5: "two",
    6: "white",
    7: "and",
    8: "one",
    9: "black",
    10: "and",
    11: "white",
    12: ",",
    13: "on",
    14: "a",
    15: "sidewalk",
    16: ".",
    17: "[SEP]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_50 = {
    "50_1": {"ids": [2], "text": "small"},
    "50_2": {"ids": [3], "text": "dogs"},
    "50_3": {"ids": [1], "text": "three"},
    "50_4": {"ids": [2, 3], "text": "small dogs"},
    "50_5": {"ids": [1, 2, 3], "text": "three small dogs"},
    "50_6": {"ids": [6], "text": "white"},
    "50_7": {"ids": [9], "text": "black"},
    "50_8": {"ids": [9, 10, 11], "text": "black and white"},
    "50_9": {"ids": [15], "text": "sidewalk"},
    "50_10": {
        "ids": [5, 6, 7, 8, 9, 10, 11],
        "text": "two white and one black and white",
    },
    "50_11": {
        "ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "text": "three small dogs, two white and one black and white",
    },
}

target_ids_250 = {
    0: "[CLS]",
    1: "two",
    2: "boys",
    3: ",",
    4: "two",
    5: "girls",
    6: ",",
    7: "strapped",
    8: "in",
    9: "and",
    10: "ready",
    11: "for",
    12: "an",
    13: "amusement",
    14: "park",
    15: "ride",
    16: ".",
    17: "[SEP]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}
instance_text_target_ids_250 = {
    "250_1": {"ids": [1], "text": "two"},
    "250_2": {"ids": [2], "text": "boys"},
    "250_3": {"ids": [5], "text": "girls"},
    "250_4": {"ids": [7], "text": "strapped"},
    "250_5": {"ids": [13], "text": "amusement"},
    "250_6": {"ids": [14], "text": "park"},
    "250_7": {"ids": [15], "text": "ride"},
    "250_8": {"ids": [1, 2], "text": "two boys"},
    "250_9": {"ids": [4, 5], "text": "two girls"},
    "250_10": {"ids": [1, 2, 3, 4, 5], "text": "two boys, two girls"},
    "250_11": {"ids": [13, 14], "text": "amusement park"},
    "250_12": {"ids": [14, 15], "text": "park ride"},
    "250_13": {"ids": [13, 14, 15], "text": "amusement park ride"},
}

target_ids_300 = {
    0: "[CLS]",
    1: "a",
    2: "young",
    3: "boy",
    4: "wearing",
    5: "a",
    6: "black",
    7: "shirt",
    8: "and",
    9: "brown",
    10: "pants",
    11: "practices",
    12: "jumping",
    13: "over",
    14: "a",
    15: "low",
    16: "bar",
    17: "on",
    18: "a",
    19: "skate",
    20: "##board",
    21: ".",
    22: "[SEP]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_300 = {
    "300_1": {"ids": [2], "text": "young"},
    "300_2": {"ids": [3], "text": "boy"},
    "300_3": {"ids": [6], "text": "black"},
    "300_4": {"ids": [7], "text": "shirt"},
    "300_5": {"ids": [9], "text": "brown"},
    "300_6": {"ids": [10], "text": "pants"},
    "300_7": {"ids": [12], "text": "jumping"},
    "300_8": {"ids": [16], "text": "bar"},
    "300_9": {"ids": [15, 16], "text": "low bar"},
    "300_10": {"ids": [19, 20], "text": "skateboard"},
    "300_11": {"ids": [2, 3], "text": "young boy"},
    "300_12": {"ids": [6, 7], "text": "black shirt"},
    "300_13": {"ids": [9, 10], "text": "brown pants"},
    "300_14": {"ids": [6, 7, 8, 9, 10], "text": "black shirt and brown pants"},
    "300_15": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "text": "young boy wearing a black shirt and brown pants",
    },
}

target_ids_400 = {
    0: "[CLS]",
    1: "a",
    2: "woman",
    3: "in",
    4: "a",
    5: "jean",
    6: "jacket",
    7: "and",
    8: "black",
    9: "sunglasses",
    10: "stands",
    11: "outside",
    12: "with",
    13: "two",
    14: "young",
    15: "boys",
    16: "by",
    17: "a",
    18: "ki",
    19: "##os",
    20: "##k",
    21: ",",
    22: "looking",
    23: "at",
    24: "a",
    25: "paper",
    26: "she",
    27: "is",
    28: "holding",
    29: "in",
    30: "her",
    31: "hand",
    32: ".",
    33: "[SEP]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_400 = {
    "400_1": {"ids": [2], "text": "woman"},
    "400_2": {"ids": [5], "text": "jean"},
    "400_3": {"ids": [6], "text": "jacket"},
    "400_4": {"ids": [8], "text": "black"},
    "400_5": {"ids": [9], "text": "sunglasses"},
    "400_6": {"ids": [13], "text": "two"},
    "400_7": {"ids": [14], "text": "young"},
    "400_8": {"ids": [15], "text": "boys"},
    "400_9": {"ids": [18, 19, 20], "text": "kiosk"},
    "400_10": {"ids": [5, 6], "text": "jean jacket"},
    "400_11": {"ids": [8, 9], "text": "black sunglasses"},
    "400_12": {"ids": [13, 14, 15], "text": "two young boys"},
    "400_13": {"ids": [2, 3, 4, 5, 6], "text": "woman in a jean jacket"},
    "400_14": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9],
        "text": "woman in a jean jacket and black sunglasses",
    },
    "400_15": {"ids": [25], "text": "paper"},
    "400_16": {"ids": [31], "text": "hand"},
}

target_ids_600 = {
    0: "[CLS]",
    1: "a",
    2: "hooded",
    3: "individual",
    4: "with",
    5: "an",
    6: "orange",
    7: "scarf",
    8: "and",
    9: "face",
    10: "covering",
    11: "uses",
    12: "a",
    13: "small",
    14: "knife",
    15: "to",
    16: "sc",
    17: "##ul",
    18: "##pt",
    19: "a",
    20: "piece",
    21: "of",
    22: "ice",
    23: ".",
    24: "[SEP]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_600 = {
    "600_1": {"ids": [2], "text": "hooded"},
    "600_2": {"ids": [3], "text": "individual"},
    "600_3": {"ids": [2, 3], "text": "hooden individual"},
    "600_4": {"ids": [6], "text": "orange"},
    "600_5": {"ids": [7], "text": "scarf"},
    "600_6": {"ids": [9], "text": "face"},
    "600_7": {"ids": [10], "text": "covering"},
    "600_8": {"ids": [14], "text": "knife"},
    "600_9": {"ids": [16, 17, 18], "text": "sculpt"},
    "600_10": {"ids": [20], "text": "piece"},
    "600_11": {"ids": [22], "text": "ice"},
    "600_12": {"ids": [6, 7], "text": "orange scarf"},
    "600_13": {"ids": [9, 10], "text": "face covering"},
    "600_14": {"ids": [20, 21, 22], "text": "piece of ice"},
    "600_15": {
        "ids": [2, 3, 4, 5, 6, 7],
        "text": "hooden individual with an orange scarf",
    },
    "600_16": {
        "ids": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "text": "hooden individual with an orange scarf and face covering",
    },
    "600_17": {"ids": [13, 14], "text": "small knife"},
}

target_ids_700 = {
    0: "[CLS]",
    1: "guy",
    2: "in",
    3: "jeans",
    4: "and",
    5: "black",
    6: "jacket",
    7: "walking",
    8: "along",
    9: "grass",
    10: "and",
    11: "trees",
    12: "with",
    13: "the",
    14: "city",
    15: "in",
    16: "the",
    17: "background",
    18: ".",
    19: "[SEP]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_700 = {
    "700_1": {"ids": [1], "text": "guy"},
    "700_2": {"ids": [3], "text": "jeans"},
    "700_3": {"ids": [5], "text": "black"},
    "700_4": {"ids": [6], "text": "jacket"},
    "700_5": {"ids": [9], "text": "grass"},
    "700_6": {"ids": [11], "text": "trees"},
    "700_7": {"ids": [14], "text": "city"},
    "700_8": {"ids": [17], "text": "background"},
    "700_9": {"ids": [1, 2, 3], "text": "guy in jeans"},
    "700_10": {"ids": [5, 6], "text": "black jacket"},
    "700_11": {"ids": [9, 10, 11], "text": "grass and trees"},
    "700_12": {"ids": [14, 15, 16, 17], "text": "city in the background"},
    "700_13": {"ids": [1, 2, 3, 4, 5, 6], "text": "guy in jeans and black jacket"},
}

target_ids_800 = {
    0: "[CLS]",
    1: "five",
    2: "children",
    3: "spin",
    4: "around",
    5: "on",
    6: "a",
    7: "playground",
    8: "roundabout",
    9: ";",
    10: "three",
    11: "lay",
    12: "on",
    13: "their",
    14: "backs",
    15: ",",
    16: "while",
    17: "one",
    18: "attempts",
    19: "to",
    20: "pull",
    21: "himself",
    22: "up",
    23: "with",
    24: "both",
    25: "arms",
    26: ",",
    27: "and",
    28: "another",
    29: "holds",
    30: "onto",
    31: "the",
    32: "side",
    33: "while",
    34: "sitting",
    35: "up",
    36: ".",
    37: "[SEP]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_800 = {
    "800_1": {"ids": [1], "text": "five"},
    "800_2": {"ids": [2], "text": "children"},
    "800_3": {"ids": [7], "text": "playground"},
    "800_4": {"ids": [8], "text": "roundabout"},
    "800_5": {"ids": [10], "text": "three"},
    "800_6": {"ids": [14], "text": "backs"},
    "800_7": {"ids": [25], "text": "arms"},
    "800_8": {"ids": [32], "text": "side"},
    "800_9": {"ids": [34], "text": "sitting"},
    "800_10": {"ids": [1, 2], "text": "five children"},
    "800_11": {"ids": [7, 8], "text": "playground roundabout"},
    "800_12": {"ids": [10, 11, 12, 13, 14], "text": "three lay on their backs"},
    "800_13": {
        "ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "text": "five children spin around on a playground roundabout",
    },
    "800_14": {
        "ids": [17, 18, 19, 20, 21, 22, 23, 24, 25],
        "text": "one attempts to pull himself up with both arms",
    },
    "800_15": {
        "ids": [28, 29, 30, 31, 32, 33, 34],
        "text": "another holds onto the side while sitting up",
    },
}

target_ids_900 = {
    0: "[CLS]",
    1: "a",
    2: "football",
    3: "player",
    4: "preparing",
    5: "a",
    6: "football",
    7: "for",
    8: "a",
    9: "field",
    10: "goal",
    11: "kick",
    12: ",",
    13: "while",
    14: "his",
    15: "teammates",
    16: "can",
    17: "coach",
    18: "watch",
    19: "him",
    20: ".",
    21: "[SEP]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_900 = {
    "900_1": {"ids": [2], "text": "football"},
    "900_2": {"ids": [3], "text": "player"},
    "900_3": {"ids": [6], "text": "football"},
    "900_4": {"ids": [9, 10], "text": "field goal"},
    "900_5": {"ids": [11], "text": "kick"},
    "900_6": {"ids": [9, 10, 11], "text": "field goal kick"},
    "900_6": {"ids": [15], "text": "teammates"},
    "900_7": {"ids": [17], "text": "coach"},
    "900_8": {"ids": [19], "text": "him"},
    "900_9": {"ids": [15, 16, 17, 18, 19], "text": "teammates can coach watch him"},
    "900_10": {
        "ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "text": "a football player preparing a football for a field goal kick",
    },
}

target_ids_1000 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_1000 = {
    "1000_1": {"ids": [2], "text": "group"},
    "1000_2": {"ids": [4], "text": "woman"},
    "1000_3": {"ids": [13], "text": "marathon"},
    "1000_4": {"ids": [6, 7, 8], "text": "various ethnic backgrounds"},
    "1000_5": {"ids": [2, 3, 4], "text": "group of woman"},
    "1000_6": {
        "ids": [2, 3, 4, 5, 6, 7, 8],
        "text": "group of woman from various ethnic backgrounds",
    },
}

target_ids_350 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_350 = {
    "350_1": {"ids": [2], "text": "group"},
}

target_ids_450 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_450 = {
    "450_1": {"ids": [2], "text": "group"},
}

target_ids_550 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_550 = {
    "550_1": {"ids": [2], "text": "group"},
}

target_ids_650 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_650 = {
    "650_1": {"ids": [2], "text": "group"},
}

target_ids_750 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_750 = {
    "750_1": {"ids": [2], "text": "group"},
}

target_ids_850 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_850 = {
    "850_1": {"ids": [2], "text": "group"},
}

target_ids_950 = {
    0: "[CLS]",
    1: "a",
    2: "group",
    3: "of",
    4: "woman",
    5: "from",
    6: "various",
    7: "ethnic",
    8: "backgrounds",
    9: "are",
    10: "competing",
    11: "in",
    12: "a",
    13: "marathon",
    14: ".",
    15: "[SEP]",
    16: "[PAD]",
    17: "[PAD]",
    18: "[PAD]",
    19: "[PAD]",
    20: "[PAD]",
    21: "[PAD]",
    22: "[PAD]",
    23: "[PAD]",
    24: "[PAD]",
    25: "[PAD]",
    26: "[PAD]",
    27: "[PAD]",
    28: "[PAD]",
    29: "[PAD]",
    30: "[PAD]",
    31: "[PAD]",
    32: "[PAD]",
    33: "[PAD]",
    34: "[PAD]",
    35: "[PAD]",
    36: "[PAD]",
    37: "[PAD]",
    38: "[PAD]",
    39: "[PAD]",
}

instance_text_target_ids_950 = {
    "950_1": {"ids": [2], "text": "group"},
}


id_to_tids = {
    # 50: instance_text_target_ids_50,
    # 100: instance_text_target_ids_100,
    # 150: instance_text_target_ids_150,
    # 200: instance_text_target_ids_200,
    # 500: instance_text_target_ids_500,
    # 250: instance_text_target_ids_250,
    # 300: instance_text_target_ids_300,
    # 400: instance_text_target_ids_400,
    # 600: instance_text_target_ids_600,
    # 700: instance_text_target_ids_700,
    # 800: instance_text_target_ids_800,
    # 900: instance_text_target_ids_900,
    # 1000: instance_text_target_ids_1000,
}

key_to_logits = {}


def find_top_k_bounding_boxes(id_to_boxes, pixel_grads, num_gt_boxes):
    top_k_box_ids = []

    box_id_to_avg_grad = {}
    for box_id, boxes in id_to_boxes.items():
        box_id_means = []
        for box_iter in boxes:
            x1, y1, x2, y2 = box_iter
            box_id_means.append(np.mean(pixel_grads[y1:y2, x1:x2]))
        box_id_to_avg_grad[box_id] = np.mean(box_id_means)

    # Sort dictionary box_id_to_avg_grad by value
    sorted_box_id_to_avg_grad = sorted(
        box_id_to_avg_grad.items(), key=lambda kv: kv[1], reverse=True
    )
    for i in range(num_gt_boxes):
        top_k_box_ids.append(sorted_box_id_to_avg_grad[i][0])
    return top_k_box_ids


for instance_idx, tid_dict in id_to_tids.items():
    key_to_logits[str(instance_idx)] = {}
    for key, value in tid_dict.items():
        key_to_logits[str(instance_idx)][key] = {}

        # Get the Instance
        instance = data.getdata(instance_idx)
        (
            first_sentence,
            id_to_boxes,
            id_to_phrase,
        ) = data.get_entities_data_first_sentence(instance_idx)
        # print(id_to_boxes, id_to_phrase)

        # Get Original Logits
        original_probs, _ = analysismodel.forward(instance)
        original_logits = original_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["original_logits"] = original_logits

        # Calculate the Double Grad
        print(instance_idx)
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-flickr30k"
        )

        grads, di, tids = analysismodel.getdoublegrad(
            instance, instance[-1], value["ids"]
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

        grads = grads[0]

        # Save the Double Grad Image
        normalized_grads_dg = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)

        heatmap2d(
            normalized_grads_dg,
            f"visuals/flickr30k-vilt-{key}-doublegrad.png",
            instance[0],
        )

        # Get the New Text
        new_tids = tids[0].detach().cpu().numpy().tolist()
        new_tids = new_tids[: value["ids"][0]] + new_tids[value["ids"][-1] + 1 :]
        sep_index = new_tids.index(processor.tokenizer.sep_token_id)

        new_text = processor.tokenizer.decode(new_tids[1:sep_index])

        # Save new text in a file
        with open(f"visuals/flickr30k-vilt-{key}-new_text.txt", "w") as f:
            f.write(new_text)

        # Load and resize original image
        normalized_grads = (
            normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
            .detach()
            .cpu()
            .numpy()
        )
        img = cv2.resize(
            np.asarray(Image.open(instance[0])),
            (normalized_grads.shape[1], normalized_grads.shape[0]),
        )

        gt_img = copy.deepcopy(img)
        random_box_img = copy.deepcopy(img)
        new_box_img = copy.deepcopy(img)

        # Ground Truth Box Drop
        # drop ground truth based on Flickr30k Entities
        # Find double grad text object

        boxes_to_drop = []
        for idx, phrase in id_to_phrase.items():
            # Check if there is an intersection between value["text"] and phrase
            if (
                value["text"].lower() in phrase.lower()
                or phrase.lower() in value["text"].lower()
            ):
                boxes_to_drop.append(idx)

        # drop boxes in image
        num_gt_boxes = 0
        gt_box_ids = []
        mask = np.zeros(gt_img.shape[:-1])
        for box_id in boxes_to_drop:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    gt_img[y1:y2, x1:x2] = 0
                    mask[y1:y2, x1:x2] = 1
                gt_box_ids.append(box_id)
                num_gt_boxes += 1
            else:
                print("Couldn't find box with box_id: ", box_id)
        gt_img_path = f"visuals/flickr30k-vilt-{key}-gt_img.jpg"

        plt.imsave(gt_img_path, gt_img)

        new_instance = (gt_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["ground_truth_logits"] = new_logits

        # Find matching boxes in img
        dg_box_ids = find_top_k_bounding_boxes(
            id_to_boxes, normalized_grads, num_gt_boxes
        )
        for box_id in dg_box_ids:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    new_box_img[y1:y2, x1:x2] = 0

        new_box_img_path = f"visuals/flickr30k-vilt-{key}-new_box_img.jpg"
        plt.imsave(new_box_img_path, new_box_img)

        print("Key: ", key)
        print(gt_box_ids, dg_box_ids)
        new_instance = (new_box_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["doublegrad_box_logits"] = new_logits

        # Randomly select num_gt_boxes from the ids and drop them

        random_box_ids = random.sample(id_to_boxes.keys(), num_gt_boxes)

        for box_id in random_box_ids:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    random_box_img[y1:y2, x1:x2] = 0

        random_box_img_path = f"visuals/flickr30k-vilt-{key}-random_box_img.jpg"
        plt.imsave(random_box_img_path, random_box_img)

        new_instance = (random_box_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["random_box_logits"] = new_logits

        num_dg_matching_boxes = len([x for x in gt_box_ids if x in dg_box_ids])
        num_random_matching_boxes = len([x for x in gt_box_ids if x in random_box_ids])
        key_to_logits[str(instance_idx)][key][
            "num_dg_matching_boxes"
        ] = num_dg_matching_boxes
        key_to_logits[str(instance_idx)][key][
            "num_random_matching_boxes"
        ] = num_random_matching_boxes
        key_to_logits[str(instance_idx)][key]["num_gt_boxes"] = num_gt_boxes


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# Write key_to_logits to JSON file
with open("key_to_logits-box-acc.json", "w") as f:
    json.dump(key_to_logits, f, cls=NumpyFloatValuesEncoder)
