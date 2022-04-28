import sys
import os

sys.path.insert(1, os.getcwd())

from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_clip import Flickr30KClip
from visualizations.visualizegradient import *
from analysis.gradientbased import get_saliency_map

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 3

# get the model
analysismodel = Flickr30KClip(target_idx=3)

instance = data.getdata(0)

# get the model predictions
preds = analysismodel.forward(instance)

# compute and print grad saliency with and without multiply orig:
saliency = get_saliency_map(instance, analysismodel, 0)
grads = saliency[0]
t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
heatmap2d(t, "visuals/flickr30k-clip-0-3-saliency.png", instance[0])
