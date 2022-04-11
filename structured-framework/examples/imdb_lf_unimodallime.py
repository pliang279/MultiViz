import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.imdb import IMDBDataset_pd
from models.imdb_raw_lf import IMDb_LF
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
datas = IMDBDataset_pd('val')
# get the model
analysismodel = IMDb_LF()
# pick data instance you want to explain
instance = datas.getdata(0)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)


# generate lime explanation for image modality on both correct label and predicted label
# explanation1 = rununimodallime(instance,'image','image',analysismodel,(predlabel,correctlabel))
# generate lime explanation for text modality on both correct label and predicted label
# explanation2 = rununimodallime(instance,'text','text',analysismodel,(predlabel,correctlabel),class_names=datas.classnames())
# visualize explanations and save to directory
# visualizelime(explanation1,'image',predlabel,'visuals/vqa-lxmert-0-image-lime-pred.png')
# visualizelime(explanation1,'image',correctlabel,'visuals/vqa-lxmert-0-image-lime-correct.png')
# visualizelime(explanation2,'text',predlabel,'visuals/vqa-lxmert-0-text-lime-pred.png')
# visualizelime(explanation2,'text',correctlabel,'visuals/vqa-lxmert-0-text-lime-correct.png')

