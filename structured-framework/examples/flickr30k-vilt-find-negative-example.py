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

import matplotlib.pyplot as plt
import matplotlib.patches as patches

random.seed(42)
# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

models = []
# get the model
for i in range(5):
    analysismodel = Flickr30KVilt(target_idx=i, device="cuda")
    models.append(analysismodel)

all_logit_scores = {}


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)



for instance_idx in range(data.length()):
    all_logit_scores[str(instance_idx)] = {}
    instance = data.getdata(instance_idx)
    # Get the Instance
    for model_idx, model in enumerate(models):
        # Get Original Logits
        original_probs, _ = model.forward(instance)
        original_logits = original_probs.detach().cpu().numpy()[0]
        all_logit_scores[str(instance_idx)][f"original_logits_{model_idx}"] = original_logits

            
# Write key_to_logits to JSON file
with open(f"all-logit-scores.json", "w") as f:
    json.dump(all_logit_scores, f, cls=NumpyFloatValuesEncoder)
