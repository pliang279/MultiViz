import os
import sys
import h5py
import torch
from gensim.models import KeyedVectors
import torchvision
import numpy as np

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_lf import IMDb_LF
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

table_path = '/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5'
dataset = h5py.File(table_path, 'r')
raw_data_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
model_path = '/home/paul/nihalj/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_best_lf_vgg_torch.pth'
multibench_path = '/home/paul/nihalj/MultiBench'
word2vec_path = '/home/paul/nihalj/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz'
raw_imdb_root_path = '/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()

# get the dataset
datas = IMDBDataset('val', raw_data_path, dataset=dataset)
# get the model
analysismodel = IMDb_LF(model_path, multibench_path, word2vec, vgg16_model, device, batch_size=32)


np.random.seed(10)
for i in np.random.randint(low=0, high=datas.length(), size=10):

    instance = datas.getdata(i)

    predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
    correctlabel = analysismodel.getcorrectlabel(instance)

    classnames = datas.classnames()
    print(f'Example {i} -- Predicted: {classnames[predlabel]}, True: {classnames[correctlabel]}')

    explanation1 = rununimodallime(instance, 'image', 'image', analysismodel, (predlabel,correctlabel))
    explanation2 = rununimodallime(instance, 'text', 'text', analysismodel, (predlabel,correctlabel), class_names=datas.classnames())

    visualizelime(explanation1, 'image', predlabel, f'visuals/imdb/imdb-lf-{i}-image-lime-pred.png')
    visualizelime(explanation1, 'image', correctlabel, f'visuals/imdb/imdb-lf-{i}-image-lime-correct.png')
    visualizelime(explanation2,'text', predlabel, f'visuals/imdb/imdb-lf-{i}-text-lime-pred.png')
    visualizelime(explanation2,'text', correctlabel, f'visuals/imdb/imdb-lf-{i}-text-lime-correct.png')

#527
#2042
#1949