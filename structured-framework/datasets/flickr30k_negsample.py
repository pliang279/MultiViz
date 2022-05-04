import os
import sys

sys.path.insert(1, os.getcwd())

import jsonlines
import random
from PIL import Image
import matplotlib.pyplot as plt

# def download_data(typ="valid"):
#     # TODO: Check if we want to allow image downloading
#     # Currently, we need to fill a form to download image data
#     os.system("bash datasets/flickr30k_extras/flickr30kdownload" + typ + "annot.sh")


class Flickr30kNegsampleDataset:
    def __init__(self, split="valid", img_dir=""):
        self.split = split
        self.img_dir = img_dir
        with jsonlines.open(f"data/flickr30k/{split}_negative_ann.jsonl") as reader:
            self.annotations = [line for line in reader]

    def getdata(self, data_idx):
        annot = self.annotations[data_idx]
        imgfile = annot["imgfile"]
        # example_idx = annot['idx']
        label = annot["label"]
        sentence = annot["sentences"]
        return imgfile, sentence, label

    def length(self):
        return len(self.annotations)

    # TODO: Check what should be done with class names
    def classnames(self):
        return None

    def sample(self, num):
        sampled = []
        nums = []
        for i in range(self.length()):
            nums.append(i)
        random.shuffle(nums)
        idx = 0
        while len(sampled) < num:
            a = self.getdata(nums[idx])
            sampled.append(a)
            idx += 1
        return sampled

    def getseqdata(self, start, end):
        a = []
        for i in range(start, end):
            a.append(self.getdata(i))
        return a

    def makepic(self, data_idx):
        data_point = self.getdata(data_idx)
        imgfile, sentence, label = data_point

        title = (
            "Caption: "
            + sentence
            + "\n Data Idx: "
            + str(data_idx)
            + "Label: "
            + str(label)
        )

        plt.clf()
        plt.imshow(Image.open(imgfile))
        plt.title(title)
        plt.savefig(
            "visuals/data/flickr30k-neg" + self.split + "-" + str(data_idx) + ".png"
        )
