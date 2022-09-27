import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.mimic import MIMICDataset
from models.mimic_lf import MIMICLF
from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime

# get the dataset
datas = MIMICDataset("test")
# get the model
analysismodel = MIMICLF("/home/anon/MultiBench/mimiclfbest.pt", "/home/anon/MultiBench")
# pick data instance you want to explain
instance = datas.getdata(0)
# get the model predicted label
predlabel = analysismodel.getpredlabel(analysismodel.forward(instance))
# get the correct label
correctlabel = analysismodel.getcorrectlabel(instance)
# generate lime explanation for tabular modality on both correct label and predicted label
explanation1 = rununimodallime(
    instance,
    "static",
    "tabular",
    analysismodel,
    (predlabel, correctlabel),
    tabularbase=datas.allstatic(),
    feature_names=["0", "1", "2", "3", "4"],
)
# generate lime explanation for timeseries modality on both correct label and predicted label
explanation2 = rununimodallime(
    instance, "timeseries", "timeseriesC", analysismodel, (predlabel, correctlabel)
)
# visualize explanations and save to directory
visualizelime(
    explanation1, "tabular", predlabel, "visuals/mimic-lf-0-tabular-lime-pred.png"
)
visualizelime(
    explanation1, "tabular", correctlabel, "visuals/mimic-lf-0-tabular-lime-correct.png"
)
visualizelime(
    explanation2,
    "timeseriesC",
    predlabel,
    "visuals/mimic-lf-0-timeseries-lime-pred.png",
)
visualizelime(
    explanation2,
    "timeseriesC",
    correctlabel,
    "visuals/mimic-lf-0-timeseries-lime-correct.png",
)
