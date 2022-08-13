import os
import sys
import torch
import torchvision
import h5py
import random
import numpy as np
from lime.wrappers.scikit_image import SegmentationAlgorithm

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_vgg_bert_lrtf import IMDb_LF
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime
from transformers import BertTokenizer, BertModel
from datasets.imdb_raw import IMDBDataset
from visualizations.visualizegradient import *
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *

random.seed(0)
np.random.seed(0)

table_path = '/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5'
dataset = h5py.File(table_path, 'r')
model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_lrtf.pth'
multibench_path = '/home/paul/nihalj/MultiBench'
raw_imdb_root_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# get the dataset
datas = IMDBDataset('val', raw_imdb_root_path, dataset=dataset, crop=False)
# get the model
analysismodel = IMDb_LF(model_path, multibench_path, bert_model, bert_tokenizer, vgg16_model, device, batch_size=32)
analysismodel.model.eval()

segmentation_fn = SegmentationAlgorithm(
    "felzenszwalb",
    scale=750, 
    sigma=0.5, 
    min_size=50
)


np.random.seed(10)
for i in np.random.randint(low=0, high=datas.length(), size=10):

    instance = datas.getdata(i)

    predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
    correctlabel = analysismodel.getcorrectlabel(instance)

    classnames = datas.classnames()
    print(
        f"Example {i} -- Predicted: {classnames[predlabel]}, True: {classnames[correctlabel]}"
    )

    explanation1 = rununimodallime(
        instance, "image", "image", analysismodel, (predlabel, correctlabel)
    )
    explanation2 = rununimodallime(
        instance,
        "text",
        "text",
        analysismodel,
        (predlabel, correctlabel),
        class_names=datas.classnames(),
    )

    visualizelime(
        explanation1,
        "image",
        predlabel,
        f"visuals/imdb/imdb-lrtf-{i}-image-lime-pred.png",
    )
    visualizelime(
        explanation1,
        "image",
        correctlabel,
        f"visuals/imdb/imdb-lrtf-{i}-image-lime-correct.png",
    )
    visualizelime(
        explanation2, "text", predlabel, f"visuals/imdb/imdb-lrtf-{i}-text-lime-pred.png"
    )
    visualizelime(
        explanation2,
        "text",
        correctlabel,
        f"visuals/imdb/imdb-lrtf-{i}-text-lime-correct.png",
    )
