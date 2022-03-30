import lime
from lime import lime_image,lime_text,lime_tabular
import numpy as np
import torch
import torch.nn.functional as F

def rununimodallime(datainstance,modalityname,modalitytype,analysismodel,labels,num_samples=100, batch_size=5, on_sparse=False, post_softmax=False, class_names = None, categorical_names = None):
    originstance = analysismodel.getunimodaldata(datainstance, modalityname)
    def classify(inputs):
        modifiedinputs = [analysismodel.replaceunimodaldata(datainstance,modalityname,i) for i in inputs]
        results = analysismodel.forwardbatch(modifiedinputs)
        if on_sparse:
            logits = [analysismodel.getprelinear(result) for result in results]
        elif post_softmax:
            logits = [F.softmax(analysismodel.getlogit(result)) for result in results]
        else:
            logits = [analysismodel.getlogit(result) for result in results]
        try:
            return [logit.cpu().detach().numpy() for logit in logits]
        except:
            return logits
    additionalparam={}
    if modalitytype == 'image':
        lime_explainer = lime_image.LimeImageExplainer()
        additionalparam['hide_color']=0
    elif modalitytype == 'text':
        lime_explainer = lime_text.LimeTextExplainer(class_names = class_names)
    elif modalitytype == 'tabular':
        lime_explainer = lime_tabular.LimeTabularExplainer(class_names = class_names, categorical_names = None)
    elif modalitytype == 'timeseries':
        lime_explainer = LimeTSExplainer()
    else:
        raise NotImplemented
    return lime_explainer.explain_instance(originstance,classify,num_samples = num_samples, batch_size = batch_size, labels = labels, **additionalparam)


        

