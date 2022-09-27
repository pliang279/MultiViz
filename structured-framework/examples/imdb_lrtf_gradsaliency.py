import os
import sys
import torch
import torchvision
import h5py
import random
import numpy as np

sys.path.insert(1,os.getcwd())

from datasets.imdb_raw import IMDBDataset
from models.imdb_raw_vgg_bert_lrtf import IMDb_LF
from transformers import BertTokenizer, BertModel
from datasets.imdb_raw import IMDBDataset
from visualizations.visualizegradient import *
from analysis.SparseLinearEncoding import *
from visualizations.visualizesparselinearmodel import *

random.seed(0)
np.random.seed(0)

table_path = '/home/anon/anon/MultiBench/multimodal_imdb.hdf5'
dataset = h5py.File(table_path, 'r')
model_path = '/home/anon/anon/multimodal_analysis/structured-framework/models/mmimdb_lf_extras/imdb_lrtf.pth'
multibench_path = '/home/anon/anon/MultiBench'
raw_imdb_root_path = '/home/anon/anon/MultiBench/datasets/imdb/mmimdb/dataset'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# get the model
analysismodel = IMDb_LF(model_path, multibench_path, bert_model, bert_tokenizer, vgg16_model, device, batch_size=32)
analysismodel.model.eval()

# get the dataset
train_dataset = IMDBDataset("train", raw_imdb_root_path, dataset=dataset, crop=False)
val_dataset = IMDBDataset("val", raw_imdb_root_path, dataset=dataset, crop=False)
test_dataset = IMDBDataset("test", raw_imdb_root_path, dataset=dataset, crop=False)

# pick data instance you want to explain
instance = train_dataset.getdata(0)

# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)

# gradients with respect to predicted label
model_inp, grads = analysismodel.getgrad(instance, predlabel)

# image gradient with respect to predicted label
t = normalize255(torch.sum(torch.abs(grads[-1]), dim=0))
tmp_path = f'visuals/imdb_lrtf_json/{instance}/{instance}.png'
plt.imsave(tmp_path, instance[1])
heatmap2d(t, f'visuals/imdb_lrtf_json/{instance}/gs{str(instance)}_pred.png', tmp_path, alpha=0.4, fivebyfive=True)

# text gradient with respect to predicted label
encoded_text = bert_tokenizer(instance[0], return_tensors='pt', truncation=True)
raw_text = bert_tokenizer.decode(encoded_text['input_ids'][0])
token_ids = encoded_text['input_ids'][0][1:-1].numpy() # remove [CLS] and [SEP] tokens
token_grads = grads[0][1:-1, :]
token_embeds = model_inp[0][1:-1, :]
saliency_grad = torch.sum(torch.abs(token_grads), dim=1).numpy()
token_grad_list = [(token_ids[i], saliency_grad[i]) for i in range(len(saliency_grad))]
token_grad_list = sorted(token_grad_list, key=lambda x: x[1], reverse=True)
salient_tokens = [tgl[0] for tgl in token_grad_list]
pred_d = {bert_tokenizer.decode([tgl[0]]): tgl[1] for tgl in token_grad_list}
textmap(list(pred_d.keys()), torch.tensor(list(pred_d.values())), f'visuals/imdb_lrtf_json/{instance}/gs_text_{instance}_pred.png')

# second order gradients
# select word
token_ids = [bert_tokenizer(raw_text, return_tensors='pt')['input_ids'][0][1].item() for raw_text in raw_text.split()[0]]
# second order gradient of image wrt text
doublegrad = analysismodel.getdoublegrad(instance, correctlabel, token_ids, prelinear=False)
t = normalize255(torch.sum(torch.abs(doublegrad), dim=0))
heatmap2d(t, f'visuals/imdb_lrtf_json/{instance}/secondgs' + str(instance) + f'_pred_{raw_text.split()[0]}_final.png', tmp_path, alpha=0.3, fivebyfive=True)
torch.cuda.empty_cache()
