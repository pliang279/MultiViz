import os
import sys

sys.path.insert(1, os.getcwd())

import jsonlines
import random
from PIL import Image
import matplotlib.pyplot as plt

# TODO: Check if we want to add negative sampler
# NOTE: We need this data to clone have the flickr30kentities data
os.system("bash datasets/flickr30k_extras/flickr30kentitiesdownloadannot.sh")
from datasets.flickr30k_extras.flickr30k_entities_utils import get_sentence_data, get_annotations

def download_data(typ="valid"):
    # TODO: Check if we want to allow image downloading
    # Currently, we need to fill a form to download image data
    os.system("bash datasets/flickr30k_extras/flickr30kdownload" + typ + "annot.sh")

class Flickr30kDataset:
    def __init__(self, split="valid", img_dir='data/flickr30k/flickr30k-images'):
        self.split = split
        self.img_dir = img_dir
        with jsonlines.open(f"data/flickr30k/{split}_ann.jsonl") as reader:
            self.annotations = [line for line in reader]

        self.entities_sentences = [get_sentence_data(f'data/flickr30k/Sentences/{annot["id"]}.txt') for annot in self.annotations]
        self.entities_annotations = [get_annotations(f'data/flickr30k/Annotations/{annot["id"]}.xml') for annot in self.annotations]

    # TODO: Add method to load annotation and sentence data for given example idx
    def get_entities_data(self, example_idx):
        pass

        
    def getdata(self, data_idx):
        annot = self.annotations[data_idx]
        img_path = annot['img_path']
        # example_idx = annot['id']
        sentences = annot['sentences']
        imgfile = f"{self.img_dir}/{img_path}"
        return imgfile, sentences

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

    def makepic(self, data_idx, sentence_idx):
        data_point = self.getdata(data_idx)
        imgfile, sentences = data_point
        sentence = sentences[sentence_idx]

        title = "Caption: " + sentence + "\n Data Idx: " + str(data_idx) + " Sentence Idx: " + str(sentence_idx)

        plt.clf()
        plt.imshow(Image.open(imgfile))
        plt.title(title)
        plt.savefig("visuals/data/flickr30k-" + self.split + "-" + str(data_idx) + "-" + str(sentence_idx) + ".png")