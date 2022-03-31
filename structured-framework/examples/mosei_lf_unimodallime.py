
import os
import sys

sys.path.insert(1,os.getcwd())

from datasets.mosei import MOSEIDataset
from models.mosei_lf import MOSEILF
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
datas = MOSEIDataset('test')
# get the model
analysismodel = MOSEILF('/home/paul/yiwei/MultiBench/mosei_lf_best.pt','/home/paul/yiwei/MultiBench')
# pick data instance you want to explain
instance = datas.getdata(0)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# generate lime explanation for image modality on both correct label and predicted label
explanation1 = rununimodallime(instance,'image','timeseries',analysismodel,(predlabel,correctlabel))
# generate lime explanation for audio modality on both correct label and predicted label
explanation2 = rununimodallime(instance,'audio','timeseries',analysismodel,(predlabel,correctlabel))
# generate lime explanation for text modality on both correct label and predicted label
explanation3 = rununimodallime(instance,'text','timeseries',analysismodel,(predlabel,correctlabel))
# visualize explanations and save to directory
visualizelime(explanation1,'timeseries',predlabel,'visuals/mosei-lf-0-image-lime-pred.png')
visualizelime(explanation1,'timeseries',correctlabel,'visuals/mosei-lf-0-image-lime-correct.png')
visualizelime(explanation2,'timeseries',predlabel,'visuals/mosei-lf-0-audio-lime-pred.png')
visualizelime(explanation2,'timeseries',correctlabel,'visuals/mosei-lf-0-audio-lime-correct.png')
visualizelime(explanation3,'timeseries',predlabel,'visuals/mosei-lf-0-text-lime-pred.png')
visualizelime(explanation3,'timeseries',correctlabel,'visuals/mosei-lf-0-text-lime-correct.png')

