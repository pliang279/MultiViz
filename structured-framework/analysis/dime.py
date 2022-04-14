import numpy as np
import torch
import copy
import random
def nopreprocess(i):
    return i


def dime(fullsamples,instanceid,analysismodel,labels,samplematrix=None,modal1=0,modal2=1,preprocessor1=nopreprocess,preprocessor2=nopreprocess,num_samples=100,samplematrixsavedir='tmp/samplematrix.pt',**kwargs):
    fullsize=len(fullsamples)
    modalnames=analysismodel.getmodalitynames()
    modaltypes=analysismodel.getmodalitytypes()
    m1n=analysismodel.getmodalitynames()[modal1]
    m2n=analysismodel.getmodalitynames()[modal2]
    m1t=analysismodel.getmodalitytypes()[modal1]
    m2t=analysismodel.getmodalitytypes()[modal2]
    lsize = analysismodel.getlogitsize()
    if samplematrix is None:
        samplematrix=torch.zeros(fullsize,fullsize,lsize)
        ins=[]
        for i in range(fullsize):
            for j in range(fullsize):
                m1in = analysismodel.getunimodaldata(fullsamples[i],m1n)
                ins.append(analysismodel.replaceunimodaldata(fullsamples[j],m1n,m1in))
        outs = analysismodel.forwardbatch(ins)
        for counts in range(len(outs)):
            samplematrix[counts // fullsize, counts % fullsize]= analysismodel.getlogit(outs[counts])
        torch.save(samplematrix,samplematrixsavedir)
    template = fullsamples[instanceid]
    def classify1(self,ins):
        reps = preprocessor1(ins)
        allvals=[]
        for rep in reps:
            cpmatrix = copy.deepcopy(samplematrix)
            ins=[]
            for j in range(fullsize):
                instance = fullsamples[j]
                newinstance = analysismodel.replaceunimodaldata(instance,m1n,rep)
                ins.append(newinstance)
            reses = analysismodel.forwardbatch(ins)
            for i in range(fullsize):
                cpmatrix[instanceid][i]=analysismodel.getlogit(res)
            avg0 = torch.mean(cpmatrix,dim=1)
            avg1 = torch.mean(cpmatrix,dim=0)
            avgs = torch.mean(avg0,dim=0)
            uniout = avg0[instanceid]+avg1[instanceid]-avgs
            multiout = cpmatrix[instanceid][instanceid]
            line=[]
            for label in labels:
                line.append(uniout[label].item())
                line.append(multiout[label].item())
            allvals.append(line)
        return np.array(allvals)
    
    def classify2(self,ins):
        reps = preprocessor2(ins)
        allvals=[]
        for rep in reps:
            cpmatrix = copy.deepcopy(samplematrix)
            ins=[]
            for j in range(fullsize):
                instance = fullsamples[j]
                newinstance = analysismodel.replaceunimodaldata(instance,m2n,rep)
                ins.append(newinstance)
            reses = analysismodel.forwardbatch(ins)
            for i in range(fullsize):
                cpmatrix[i][instanceid]=analysismodel.getlogit(res)
            avg0 = torch.mean(cpmatrix,dim=1)
            avg1 = torch.mean(cpmatrix,dim=0)
            avgs = torch.mean(avg0,dim=0)
            uniout = avg0[instanceid]+avg1[instanceid]-avgs
            multiout = cpmatrix[instanceid][instanceid]
            line=[]
            for label in labels:
                line.append(uniout[label].item())
                line.append(multiout[label].item())
            allvals.append(line)
        return np.array(allvals)
    lbs = []
    for i in range(2*len(labels)):
        lbs.append(i)
    explainer1,additionalparam1 = getlimeexplainer(m1t,**kwargs)
    explainer2,additionalparam2 = getlimeexplainer(m2t,**kwargs)
    exp1=explainer1.explain_instance(analysismodel.getunimodaldata(template,m1n),classify1,num_samples = num_samples, labels = lbs, top_labels=None,**additionalparam1)
    exp2=explainer2.explain_instance(analysismodel.getunimodaldata(template,m2n),classify2,num_samples = num_samples, labels = lbs, top_labels=None,**additionalparam2)
    return exp1,exp2

    

from lime import lime_image,lime_text
from analysis.unimodallime import *
def getlimeexplainer(modalitytype,batch_size=5,class_names=None,feature_names=None,tabularbase=None,totallabels=None):
    additionalparam={}
    if modalitytype == 'image':
        lime_explainer = lime_image.LimeImageExplainer()
        additionalparam['hide_color']=0
        additionalparam['batch_size']=batch_size
    elif modalitytype == 'text':
        lime_explainer = lime_text.LimeTextExplainer(class_names = class_names)
    elif modalitytype == 'tabular':
        lime_explainer = lime_tabular.LimeTabularExplainer(class_names = class_names, feature_names = feature_names,training_data=tabularbase)
    elif modalitytype == 'timeseries':
        lime_explainer = EmbeddingTimeSeriesExplainer()
        additionalparam['totallabels'] = totallabels
    elif modalitytype == 'timeseriesC':
        lime_explainer = CategoricalTimeSeriesExplainer()
        additionalparam['totallabels'] = totallabels
    else:
        raise NotImplemented
    return lime_explainer,additionalparam










def imgfprocess(inps):
    names=[]
    for inp in inps:
        ii = random.randint(0,100000000)
        plt.imsave(f"tmp/zt{ii}.jpg",inp)
        names.append(f"tmp/zt{ii}.jpg")
    return names
