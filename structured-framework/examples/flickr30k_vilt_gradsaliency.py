import sys
import os

sys.path.insert(1, os.getcwd())

from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_vilt import Flickr30KVilt
from visualizations.visualizegradient import *
from analysis.gradientbased import get_saliency_map

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

# get the model
analysismodel = Flickr30KVilt(target_idx=target_idx)

for instance_idx in [
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
    1000,
    808,
    204,
    654,
    589,
    711,
    777,
    411,
    265,
    169,
    308,
    259,
    634,
    391,
    576
]:
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

    # copy images to directory
    os.system(
        f"cp {instance[0]} visuals/flickr30k-vilt-{instance_idx}-{target_idx}-image.png"
    )
    # save text to file
    with open(f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-text.txt", "w") as f:
        f.write(instance[1][target_idx])
