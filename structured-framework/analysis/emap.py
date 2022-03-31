import numpy as np
from analysis.utils import tryconverttonp
def emap(emapinstance,sampleinstances,modalityname1,modalityname2,analysismodel):
    numsamples = len(sampleinstances)+1
    storage = np.zeros((numsamples,numsamples,analysismodel.getlogitsize()))
    sampleinstances.append(emapinstance)
    for i in range(numsamples):
        for j in range(numsamples):
            modified = analysismodel.replaceunimodaldata(sampleinstances[i],modalityname2,analysismodel.getunimodaldata(sampleinstances[j],modalityname2))
            storage[i][j] = tryconverttonp(analysismodel.getlogit(analysismodel.forward(modified)))
    uni1 = np.mean(storage[-1],axis=0)
    uni2 = np.mean(storage[:,-1],axis=0)
    alls = np.mean(storage,axis=(0,1))
    return uni1+uni2-alls


def emap_print_report(emapinstance,sampleinstances,modalityname1,modalityname2,analysismodel,labels):
    origouts = tryconverttonp(analysismodel.getlogit(analysismodel.forward(emapinstance)))
    emapouts = emap(emapinstance,sampleinstances,modalityname1,modalityname2,analysismodel)
    for label in labels:
        print("Label "+ str(label)+" orig logit: "+ str(origouts[label])+" emap logit: "+ str(emapouts[label]))
    print("orig top label: "+str(np.argmax(origouts)))
    print("emap top label: "+str(np.argmax(emapouts)))

