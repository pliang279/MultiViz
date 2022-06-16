import matplotlib.pyplot as plt
import torch


def sparsityaccgraph(res, savedir, show=False, scatter=False):
    plt.clf()
    if scatter:
        plt.scatter(res[1], res[0])
    else:
        plt.plot(res[1], res[0])
    if show:
        plt.show()
    plt.savefig(savedir)


from analysis.unimodallime import rununimodallime
from visualizations.visualizelime import visualizelime


def analyzepointandvisualizeall(
    params,
    datainstance,
    analysismodel,
    label,
    prefix,
    prefixall,
    pathnum=95,
    k=5,
    numfeats=3,
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    zzs = []
    for i in topk:
        zz=glmres["weight"][label][i]
        print(zz)
        zzs.append(zz.item())
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]
        retters = rununimodallime(
            datainstance,
            modalityname,
            modalitytype,
            analysismodel,
            topk,
            on_sparse=True,
        )
        for j in range(k):
            visualizelime(
                retters,
                modalitytype,
                topk[j],
                prefix + "-" + modalityname + "-lime-feat" + str(topk[j]) + ".png",
                num_features=numfeats,
            )
    plt.clf()
    fig, ax = plt.subplots(
        nrows=len(analysismodel.getmodalitynames()), ncols=k, figsize=(30, 21)
    )

    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            ax[i][j].imshow(
                plt.imread(
                    prefix + "-" + modalityname + "-lime-feat" + str(topk[j]) + ".png"
                )
            )
    plt.savefig(prefixall + "-all-lime-feats.png")
    plt.close()
    return topk.tolist(),zzs


from visualizations.visualizegradient import *


def analyzepointandvisualizeallgrad(
    params, datainstance, analysismodel, label, prefix, prefixall, pathnum=95, k=5
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    for i in topk:
        print(glmres["weight"][label][i])
    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]
        # retters=rununimodallime(datainstance,modalityname,modalitytype,analysismodel,topk,on_sparse=True)
        for j in range(k):
            if modalitytype == "image":
                raw, grad, fi = analysismodel.getgrad(
                    datainstance, topk[j], prelinear=True
                )
                t = normalize255(torch.sum(torch.abs(grad), dim=0))
                heatmap2d(t, prefix + "-image-feat-" + str(topk[j]) + ".png", fi)
            elif modalitytype == "text":
                grad, sent, _, _ = analysismodel.getgradtext(
                    datainstance, topk[j], prelinear=True
                )
                textmap(sent, grad, prefix + "-text-feat-" + str(topk[j]) + ".png")

    plt.clf()
    fig, ax = plt.subplots(
        nrows=len(analysismodel.getmodalitynames()), ncols=k, figsize=(30, 21)
    )

    for i in range(len(analysismodel.getmodalitynames())):
        modalitytype = analysismodel.getmodalitytypes()[i]
        for j in range(k):
            if modalitytype == "image":
                ax[i][j].imshow(
                    plt.imread(prefix + "-image-feat-" + str(topk[j]) + ".png")
                )
            elif modalitytype == "text":
                ax[i][j].imshow(
                    plt.imread(prefix + "-text-feat-" + str(topk[j]) + ".png")
                )
    plt.savefig(prefixall + "-all-grad-feats.png")
    plt.close()


def analyzefeaturesandvisualizeallgrad(
    params,
    datainstances,
    analysismodel,
    label,
    prefix,
    prefixall,
    prelinear=None,
    pathnum=95,
    k=5,
    pointsperfeat=1,
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    idxs = []
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
            torch.save(prelinear, "plclevr.pt")

        maximal_idx = torch.argmax(prelinear, dim=0)
        for j in range(k):
            maximal_idxs = prelinear[:, topk[j]].argsort()[-pointsperfeat:]
            print(maximal_idxs)
            for jj in range(pointsperfeat):
                idxs[i].append(maximal_idxs[jj])
                datainstance = datainstances[
                    maximal_idxs[jj]
                ]  # use the most activating example for this feature
                if modalitytype == "image":
                    raw, grad, fi = analysismodel.getgrad(
                        datainstance, topk[j], prelinear=True
                    )
                    t = normalize255(torch.sum(torch.abs(grad), dim=0))
                    heatmap2d(
                        t,
                        prefix
                        + "-image-grad-feat"
                        + str(topk[j])
                        + "-"
                        + str(jj)
                        + ".png",
                        fi,
                    )
                elif modalitytype == "text":
                    grad, sent, _, _ = analysismodel.getgradtext(
                        datainstance, topk[j], prelinear=True
                    )
                    textmap(
                        sent,
                        grad,
                        prefix
                        + "-text-grad-feat"
                        + str(topk[j])
                        + "-"
                        + str(jj)
                        + ".png",
                    )
    plt.clf()
    fig, ax = plt.subplots(
        nrows=len(analysismodel.getmodalitynames()) * pointsperfeat,
        ncols=k,
        figsize=(30, 21),
    )

    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            for jj in range(pointsperfeat):
                ax[i * pointsperfeat + jj][j].imshow(
                    plt.imread(
                        prefix
                        + "-"
                        + modalityname
                        + "-grad-feat"
                        + str(topk[j])
                        + "-"
                        + str(jj)
                        + ".png"
                    )
                )
                ax[i * pointsperfeat + jj][j].title.set_text(
                    str(idxs[i][j * pointsperfeat + jj])
                )
    plt.savefig(prefixall + "-all-grad-feats.png")
    plt.close()


def onefeature(dataset,analysismodel,featnum,prefix,prefixall,datainstances,prelinear,dataprefix,k=3,numfeats=3):
    idx = torch.argsort(prelinear[:,featnum])
    minidx = idx[0:k]
    maxidx = idx[-k:]

    for i in range(len(analysismodel.getmodalitynames())):
        ids=[]
        modalityname = analysismodel.getmodalitynames()[i]
        modalitytype = analysismodel.getmodalitytypes()[i]
        for jj in range(2*k):
            if jj >= k:
                idd = maxidx[jj-k].item()
            else:
                idd = minidx[jj].item()
            ids.append(idd)
            retters = rununimodallime(
                datainstances[idd],
                modalityname,
                modalitytype,
                analysismodel,
                [featnum],
                on_sparse=True,
            )
            visualizelime(retters,modalitytype,featnum,
                prefix+"-"+str(idd)+"-"+ modalityname+ "-lime-feat"+ str(featnum)+ "-"+ str(jj)+ ".png", num_features=numfeats)
    for j in range(2*k): 
        dataset.makepic(ids[j],pr=analysismodel.getpredlabel(analysismodel.forward(datainstances[ids[j]])))
    plt.clf()
    fig, ax = plt.subplots(
        nrows=(1+len(analysismodel.getmodalitynames())),
        ncols=2*k,
        figsize=(30, 21),
    )
    for j in range(2*k):
        ax[0][j].imshow(plt.imread(dataprefix+str(ids[j])+".png")) 
        ax[0][j].title.set_text(str(ids[j]))

    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for jj in range(2*k):
            ax[i+1][jj].imshow(plt.imread(prefix+"-"+str(ids[jj])+"-"+ modalityname+ "-lime-feat"+ str(featnum)+ "-"+ str(jj)+ ".png"))
            ax[i+1][jj].title.set_text(str(ids[jj]))
    plt.savefig(prefixall + "-"+str(featnum)+"-all-examples.png")
    plt.close()





def analyzefeaturesandvisualizeall(
    params,
    datainstances,
    analysismodel,
    label,
    prefix,
    prefixall,
    prelinear=None,
    pathnum=95,
    k=5,
    numfeats=3,
    pointsperfeat=1,
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    idxs=[]
    mmidxs=[]
    for i in range(len(analysismodel.getmodalitynames())):
        mmidxs=[]
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
            maximal_idxs = prelinear[:, topk[j]].argsort()[-pointsperfeat:]
            print(maximal_idxs)
            mmidxs.append(maximal_idxs.cpu().tolist())
            for jj in range(pointsperfeat):
                idxs[i].append(maximal_idxs[jj])
                datainstance = datainstances[
                    maximal_idxs[jj]
                ]  # use the most activating example for this feature
                retters = rununimodallime(
                    datainstance,
                    modalityname,
                    modalitytype,
                    analysismodel,
                    topk,
                    on_sparse=True,
                )
                visualizelime(
                    retters,
                    modalitytype,
                    topk[j],
                    prefix
                    + "-"
                    + modalityname
                    + "-lime-feat"
                    + str(topk[j])
                    + "-"
                    + str(jj)
                    + ".png",
                    num_features=numfeats,
                )
    plt.clf()
    fig, ax = plt.subplots(
        nrows=len(analysismodel.getmodalitynames()) * pointsperfeat,
        ncols=k,
        figsize=(30, 21),
    )

    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            for jj in range(pointsperfeat):
                ax[i * pointsperfeat + jj][j].imshow(
                    plt.imread(
                        prefix
                        + "-"
                        + modalityname
                        + "-lime-feat"
                        + str(topk[j])
                        + "-"
                        + str(jj)
                        + ".png"
                    )
                )
                ax[i * pointsperfeat + jj][j].title.set_text(
                    str(idxs[i][jj * pointsperfeat + j])
                )
    plt.savefig(prefixall + "-all-lime-feats.png")
    plt.close()
    return mmidxs


def makefeats(
    topk,
    datainstances,
    analysismodel,
    prefix,
    prefixall,
    prelinear=None,
    numfeats=3,
    pointsperfeat=3,
):
    idxs = []
    k = len(topk)
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
            maximal_idxs = prelinear[:, topk[j]].argsort()[-pointsperfeat:]
            print(maximal_idxs)
            for jj in range(pointsperfeat):
                idxs[i].append(maximal_idxs[jj])
                datainstance = datainstances[
                    maximal_idxs[jj]
                ]  # use the most activating example for this feature
                pred = analysismodel.getpredlabel(analysismodel.forward(datainstance))
                print(
                    str(pred) + " " + str(analysismodel.getcorrectlabel(datainstance))
                )
                retters = rununimodallime(
                    datainstance,
                    modalityname,
                    modalitytype,
                    analysismodel,
                    topk,
                    on_sparse=True,
                )
                visualizelime(
                    retters,
                    modalitytype,
                    topk[j],
                    prefix
                    + "-"
                    + modalityname
                    + "-lime-feat"
                    + str(topk[j])
                    + "-"
                    + str(jj)
                    + ".png",
                    num_features=numfeats,
                )
    plt.clf()
    fig, ax = plt.subplots(
        nrows=len(analysismodel.getmodalitynames()) * pointsperfeat,
        ncols=k,
        figsize=(30, 21),
    )

    for i in range(len(analysismodel.getmodalitynames())):
        modalityname = analysismodel.getmodalitynames()[i]
        for j in range(k):
            for jj in range(pointsperfeat):
                ax[i * pointsperfeat + jj][j].imshow(
                    plt.imread(
                        prefix
                        + "-"
                        + modalityname
                        + "-lime-feat"
                        + str(topk[j])
                        + "-"
                        + str(jj)
                        + ".png"
                    )
                )
                ax[i * pointsperfeat + jj][j].title.set_text(
                    str(idxs[i][jj * pointsperfeat + j])
                )
    plt.savefig(prefixall + "-all-lime-feats.png")
    plt.close()
