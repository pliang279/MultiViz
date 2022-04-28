import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_clip import Flickr30KClip
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 3

# get the model
analysismodel = Flickr30KClip(target_idx=3)
# pick data instance you want to explain
instance = data.getdata(554)

# generate lime explanation for image modality on both correct label and predicted label
explanation1 = rununimodallime(
    instance, "image", "image", analysismodel, (0,)
)
# generate lime explanation for text modality on both correct label and predicted label
explanation2 = rununimodallime(
    instance,
    "text",
    "text",
    analysismodel,
    (0,),
    class_names=datas.classnames(),
)
# visualize explanations and save to directory
visualizelime(
    explanation1, "image", 0, "visuals/flickr30k-clip-554-3-image-lime-pred.png"
)
visualizelime(
    explanation2, "text", 0, "visuals/flickr30k-clip-554-3-text-lime-pred.png"
)