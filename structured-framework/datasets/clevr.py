import os
import sys

sys.path.insert(1, os.getcwd())
import json
import random
import numpy as np
import PIL
from PIL import Image
import platform

# download train, test, and val split
def download_data():
    os.system("bash datasets/clevr_extras/clevrdownload.sh")


class CLEVRDataset:
    # split = train/test/val
    def __init__(self, split="val"):
        self.answermapping = [
            "yes",
            "no",
            "small",
            "large",
            "gray",
            "red",
            "blue",
            "green",
            "brown",
            "purple",
            "cyan",
            "yellow",
            "cube",
            "sphere",
            "cylinder",
            "rubber",
            "metal",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ]
        with open("data/CLEVR_v1.0/questions/CLEVR_" + split + "_questions.json") as f:
            self.q = json.load(f)
        self.split = split

    def getdata(self, idx):
        question_info = self.q["questions"][idx]
        image_id = question_info["image_index"]
        question = question_info["question"]
        question_id = question_info["question_index"]
        answer = question_info["answer"]
        imgfile = (
            "data/CLEVR_v1.0/images/"
            + self.split
            + "/CLEVR_"
            + self.split
            #+ f"_{image_id:06d}.png"
            + "_"
            + str(image_id).zfill(6)
            + ".png"
        )
        image = np.asarray(Image.open(imgfile).convert("RGB"))
        label = self.answermapping.index(answer)
        return imgfile, question, answer, label

    def length(self):
        return len(self.q["questions"])

    def classnames(self):
        return self.answermapping

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
