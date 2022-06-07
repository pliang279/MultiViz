import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_vilt import Flickr30KVilt
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

# get the model
analysismodel = Flickr30KVilt(target_idx=target_idx)
# pick data instance you want to explain

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
]:
    instance = data.getdata(instance_idx)

    # generate lime explanation for image modality on both correct label and predicted label
    explanation1 = rununimodallime(
        instance,
        "image",
        "image",
        analysismodel,
        (0,),
        num_samples=500,
    )
    # generate lime explanation for text modality on both correct label and predicted label
    explanation2 = rununimodallime(
        instance,
        "text",
        "text",
        analysismodel,
        (0,),
        class_names=data.classnames(),
        num_samples=500,
    )
    # visualize explanations and save to directory
    visualizelime(
        explanation1,
        "image",
        0,
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-image-lime-pred.png",
    )
    visualizelime(
        explanation2,
        "text",
        0,
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-text-lime-pred.png",
    )

    # copy images to directory
    os.system(
        f"cp {instance[0]} visuals/flickr30k-vilt-{instance_idx}-{target_idx}-image.png"
    )
    # save text to file
    with open(f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-text.txt", "w") as f:
        f.write(instance[1][target_idx])
