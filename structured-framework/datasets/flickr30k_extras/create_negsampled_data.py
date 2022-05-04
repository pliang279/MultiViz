import jsonlines

import os
import random
import numpy as np
import pandas as pd

def download_data(typ="valid"):
    # TODO: Check if we want to allow image downloading
    # Currently, we need to fill a form to download image data
    os.system("bash datasets/flickr30k_extras/flickr30kdownload" + type + "annot.sh")


np.random.seed(42)
random.seed(42)

NEGSAMPLES_PER_IMAGE = 5
IMG_DIR = 'data/flickr30k/flickr30k-images'
SPLITS = ['train', 'valid', 'test']

def get_data_point(annotations, data_idx):
    annot = annotations[data_idx]
    img_path = annot['img_path']
    example_idx = annot['id']
    sentences = annot['sentences']
    imgfile = f"{IMG_DIR}/{img_path}"
    return example_idx, imgfile, sentences

def get_negative_sentence(annotations, data_idx, negsamples=NEGSAMPLES_PER_IMAGE):
    # Get NEGSAMPLES_PER_IMAGE negative data points from all the annotations except current data_idx
    negative_sentences = []
    negative_example_idxs = []
    negative_sentence_idxs = []
    negative_imgfiles = []
    for i in range(negsamples):
        neg_data_idx = random.randint(0, len(annotations)-1)
        while neg_data_idx == data_idx:
            neg_data_idx = random.randint(0, len(annotations)-1)

        # Get the negative data point
        neg_example_idx, neg_imgfile, neg_sentences = get_data_point(annotations, neg_data_idx)

        # Select a sentence from the negative data point
        neg_sentence_idx = random.randint(0, 4)

        negative_sentences.append(neg_sentences[neg_sentence_idx])
        negative_example_idxs.append(neg_example_idx)
        negative_sentence_idxs.append(neg_sentence_idx)
        negative_imgfiles.append(neg_imgfile)
    return negative_sentences, negative_example_idxs, negative_sentence_idxs, negative_imgfiles

if __name__ == "__main__":
    for SPLIT in SPLITS:
        with jsonlines.open(f"data/flickr30k/{SPLIT}_ann.jsonl") as reader:
            annotations = [line for line in reader]

        # For all data points in the annotation, create negative points and save to JSON file
        for data_idx in range(len(annotations)):
            # Get the data point
            example_idx, imgfile, sentences = get_data_point(annotations, data_idx)

            # Get the negative sentences
            negative_sentences, negative_example_idxs, negative_sentence_idxs, negative_imgfiles = get_negative_sentence(annotations, data_idx)

            # Create a dictionary for the data point
            counter = 0
            data_points = []
            # Positive Data Points
            for i in range(len(sentences)):

                data_points.append(
                    {
                    "idx": counter,
                    "image_example_idx": example_idx,
                    "sentence_example_idx": example_idx,
                    "imgfile": imgfile,
                    "sentences": sentences[i],
                    "sentence_idx": i,
                    "label": 1
                    } 
                )

                counter += 1

            # Negative Data Points
            for i in range(len(negative_sentences)):
                counter += 1

                data_points.append(
                    {
                    "idx": counter,
                    "image_example_idx": example_idx,
                    "sentence_example_idx": negative_example_idxs[i],
                    "imgfile": negative_imgfiles[i],
                    "sentences": negative_sentences[i],
                    "sentence_idx": negative_sentence_idxs[i],
                    "label": 0
                    }
                )
            
            for data_point in data_points:
                # Save the data point to JSON file
                with jsonlines.open(f"data/flickr30k/{SPLIT}_negative_ann.jsonl", mode='a') as writer:
                    writer.write(data_point)

        
