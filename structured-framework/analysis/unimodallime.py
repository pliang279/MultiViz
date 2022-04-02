import lime
from lime import lime_image,lime_text,lime_tabular
import numpy as np
import torch
import torch.nn.functional as F
from analysis.utils import tryconverttonp


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
        return np.asarray([tryconverttonp(logit) for logit in logits])
    additionalparam={}
    totallabels = analysismodel.getlogitsize()
    if on_sparse:
        totallabels = analysismodel.getprelinearsize()
    if modalitytype == 'image':
        lime_explainer = lime_image.LimeImageExplainer()
        additionalparam['hide_color']=0
        additionalparam['batch_size']=batch_size
    elif modalitytype == 'text':
        lime_explainer = lime_text.LimeTextExplainer(class_names = class_names)
    elif modalitytype == 'tabular':
        lime_explainer = lime_tabular.LimeTabularExplainer(class_names = class_names, categorical_names = None)
    elif modalitytype == 'timeseries':
        lime_explainer = EmbeddingTimeSeriesExplainer()
        additionalparam['totallabels'] = totallabels
    elif modalitytype == 'timeseriesC':
        lime_explainer = CategoricalTimeSeriesExplainer()
        additionalparam['totallabels'] = totallabels
    else:
        raise NotImplemented
    return lime_explainer.explain_instance(originstance,classify,num_samples = num_samples, labels = labels, **additionalparam)


        








from lime import lime_base
import copy
import random
from sklearn.utils import check_random_state
from scipy import spatial
class CategoricalTimeSeriesExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=False):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,labels, num_samples,totallabels, seed=0, fracs=1):
        correct = labels
        samples = num_samples
        randomstate=check_random_state(seed)
        masks=randomstate.randint(0,fracs+1,(samples)*len(inp[0])).reshape(samples,len(inp[0])).astype(np.float64)
        masks /= float(fracs)
        #print(samples)
        distances = np.zeros(samples)
        llabels=np.zeros((samples,totallabels))
        datas = np.zeros((samples,len(inp),len(inp[0])))
        for i in range(samples):
            if i==0 or (np.sum(masks[i])==0.0):
                datas[i]=inp
                distances[i]=0.0
                masks[i]=np.ones(len(inp[0]))
            else:   
                datas[i]=np.einsum("ij,j->ij",inp,masks[i])
                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            llabels[i]=classfn(datas[i])
        ret={}
        for corr in correct:
            ret[str(corr)] = self.base.explain_instance_with_data(masks,llabels,distances,corr,len(inp[0]),feature_selection=self.fs)
        return ret
            

class EmbeddingTimeSeriesExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=False):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,labels, num_samples,totallabels, seed=0, fracs=1,framelength=5):
        #print("Explaining ")
        correct = labels
        samples = num_samples
        randomstate=check_random_state(seed)
        segments=(len(inp))//framelength
        masks=randomstate.randint(0,fracs+1,(samples)*segments).reshape(samples,segments).astype(np.float64)
        masks /= float(fracs)
        #print(samples)
        distances = np.zeros(samples)
        llabels=np.zeros((samples,totallabels))
        datas = np.zeros((samples,len(inp),len(inp[0])))
        for i in range(samples):
            if i==0 or (np.sum(masks[i])==0.0):
                datas[i]=inp
                distances[i]=0.0
                masks[i]=np.ones(segments)
            else:
                #print(masks[i])
                #print(inp.shape)
                datas[i]=np.einsum("ijk,i->ijk",inp.reshape(segments,framelength,len(inp[0])),masks[i]).reshape(len(inp),len(inp[0]))
                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            #print(classfn(datas[i:i+1]))
            llabels[i]=classfn(datas[i:i+1])
        #print(datas)
        #print(labels)
        ret={}
        for corr in correct:
            ret[str(corr)] = self.base.explain_instance_with_data(masks,llabels,distances,corr,len(inp[0]),feature_selection=self.fs),framelength
        return ret
