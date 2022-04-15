import matplotlib.pyplot as plt
import torch

def sparsityaccgraph(res,savedir,show=False,scatter=False):
    plt.clf()
    if scatter:
        plt.scatter(res[1],res[0])
    else:
        plt.plot(res[1],res[0])
    if show:
        plt.show()
    plt.savefig(savedir)

from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime
def analyzepointandvisualizeall(params,datainstance,analysismodel,label,prefix,prefixall,pathnum = 95,k=5,numfeats=3):
    glmres = params['path'][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]
        retters=rununimodallime(datainstance,modalityname,modalitytype,analysismodel,topk,on_sparse=True)
        for j in range(k):
            visualizelime(retters,modalitytype,topk[j],prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png',num_features=numfeats)
    plt.clf()
    fig,ax=plt.subplots(nrows=len(analysismodel.getmodalitynames()),ncols=k,figsize=(30,21))
    
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            ax[i][j].imshow(plt.imread(prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'.png'))
    plt.savefig(prefixall+'-all-lime-feats.png')


def analyzefeaturesandvisualizeall(params, datainstances, analysismodel, label, prefix, prefixall, prelinear=None, pathnum=95, k=5,numfeats=3,pointsperfeat=1):
    glmres = params['path'][pathnum]
    topk = glmres['weight'][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    idxs=[]
    for i in range(len(analysismodel.getmodalitynames())):
        idxs.append([])
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]

        # get prelinear features if not specified already
        if prelinear is None:
            model_outs = analysismodel.forwardbatch(datainstances)
            prelinear = torch.zeros((len(model_outs), analysismodel.getprelinearsize()))
            for j, model_out in enumerate(model_outs):
                prelinear[j] = analysismodel.getprelinear(model_outs[j])

        maximal_idx = torch.argmax(prelinear, dim=0)
        for j in range(k):
            maximal_idxs = prelinear[:,topk[j]].argsort()[-pointsperfeat:]
            for jj in range(pointsperfeat):
                idxs[i].append(maximal_idxs[jj])
                datainstance = datainstances[maximal_idxs[jj]] # use the most activating example for this feature
                retters=rununimodallime(datainstance,modalityname,modalitytype,analysismodel,topk,on_sparse=True)
                visualizelime(retters,modalitytype,topk[j],prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'-'+str(jj)+'.png',num_features=numfeats)
    plt.clf()
    fig,ax=plt.subplots(nrows=len(analysismodel.getmodalitynames())*pointsperfeat,ncols=k,figsize=(30,21))
    
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            for jj in range(pointsperfeat):
                ax[i*pointsperfeat+jj][j].imshow(plt.imread(prefix+'-'+modalityname+'-lime-feat'+str(topk[j])+'-'+str(jj)+'.png'))
                ax[i*pointsperfeat+jj][j].title.set_text(str(idxs[i][j]))
    plt.savefig(prefixall+'-all-lime-feats.png')
