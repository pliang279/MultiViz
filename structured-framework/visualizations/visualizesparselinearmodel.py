import matplotlib.pyplot as plt
import torch

def sparsityaccgraph(res,savedir,show=False):
    plt.clf()
    plt.plot(res[1],res[0])
    if show:
        plt.show()
    plt.savefig(savedir)

from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime
def analyzepointandvisualizeall(params,datainstance,analysismodel,label,prefix,pathnum = 95,k=5):
    glmres = params['path'][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]
        retters=rununimodallime(datainstance,modalityname,modalitytype,analysismodel,topk,on_sparse=True)
        for j in range(k):
            visualizelime(retters,modalitytype,topk[j],prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png')
    plt.clf()
    fig,ax=plt.subplots(nrows=len(analysismodel.getmodalitynames()),ncols=k,figsize=(30,21))
    
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            ax[i][j].imshow(plt.imread(prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png'))
    plt.savefig(prefix+'-all-lime-feats.png')


def analyzefeaturesandvisualizeall(params, datainstances, analysismodel, label, prefix, prelinear=None, pathnum=95, k=5):
    glmres = params['path'][pathnum]
    topk = glmres['weight'][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]

        # get prelinear features if not specified already
        if prelinear == None:
            model_outs = analysismodel.forwardbatch(datainstances)
            prelinear = torch.zeros((len(model_outs), analysismodel.getprelinearsize()))
            for j, model_out in enumerate(model_outs):
                prelinear[j] = model_out[1]

        maximal_idx = torch.argmax(prelinear, dim=0)
        for j in range(k):
            datainstance = datainstances[maximal_idx[topk[j]]] # use the most activating example for this feature
            retters=rununimodallime(datainstance,modalityname,modalitytype,analysismodel,topk,on_sparse=True)
            visualizelime(retters,modalitytype,topk[j],prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png')
    plt.clf()
    fig,ax=plt.subplots(nrows=len(analysismodel.getmodalitynames()),ncols=k,figsize=(30,21))
    
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            ax[i][j].imshow(plt.imread(prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png'))
    plt.savefig(prefix+'-all-lime-feats.png')