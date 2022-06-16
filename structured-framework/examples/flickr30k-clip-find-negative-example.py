import gc
import sys
import os

import json

sys.path.insert(1, os.getcwd())
from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_clip import Flickr30KClip

from visualizations.visualizegradient import *
import random
from tqdm.auto import tqdm

random.seed(42)
# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

models = []
# get the model
for i in range(1):
    analysismodel = Flickr30KClip(target_idx=i, device="cuda")
    models.append(analysismodel)

all_logit_scores = {}


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)



for instance_idx in tqdm(range(data.length())):
    all_logit_scores[str(instance_idx)] = {}
    instance = data.getdata(instance_idx)
    torch.cuda.empty_cache()
    gc.collect()
    # Get the Instance
    for model_idx, model in enumerate(models):
        # Get Original Logits
        original_probs = model.forward(instance)[0]
        original_logits = original_probs.detach().cpu().numpy()[0]
        all_logit_scores[str(instance_idx)][f"original_logits_{model_idx}"] = original_logits

            
# Write key_to_logits to JSON file
with open(f"all-logit-scores-clip.json", "w") as f:
    json.dump(all_logit_scores, f, cls=NumpyFloatValuesEncoder)
