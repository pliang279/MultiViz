import matplotlib.pyplot as plt
import matplotlib
import torch
import sys
import os
import json
sys.path.insert(1,os.getcwd())
from datasets.mosei2 import*
from models.mosei_mult import*

fau_agg_dict = {
    "FaceEmotion": [0,1,2,3,4,5,6,7,8,9],
    "Brow": [10,11,12],
    "Eye": [13,14,15, 29],
    "Nose": [16],
    "Lip": [17,18,19,20,22,23,24,25,26,28],
    "Chin": [21,27],
    "HeadMovement":[32,33,34],
    "Others": [30,31]
}

fau_agg_dict_2 = {
    "FaceEmotion": [0,1,2,3,4,5,6,7,8,9],
    "Brow": [10,11,12],
    "Eye": [13,14,15, 29],
    "Nose": [16],
    "Lip": [17,18,19,20,22,23,24,25,26,28],
    "Chin": [21,27],
    "HeadMovement":[32,33,34],
    "Has_Glasses": [30],
    "Is_Male": [31]
}

fau_colors = ['r']*10+['g']*3+['b']*4+['c']+['m']*10+['y']*2+['black']*3+['orange']*2

audio_agg_dict = {
    "pitch": [0],
    "glottal": [1,2,3,4,5,6,7,8,9,10],
    "amplitude": [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    "phase": [36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73]
}
audio_colors = ['r']+['g']*10+['b']*25+['black']*38


vision_order = []
audio_order = []
for v in fau_agg_dict.values():
    vision_order += v
for v in audio_agg_dict.values():
    audio_order += v    

def visualize_grad(dataset, model, idx, reverse=False):
    #savedir = 'visuals/mosei_grad/mosei_'+str(idx)
    savedir = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    datainstance = dataset.getdata(idx)
    if not reverse:
        text_grad, _ = model.getgrad(datainstance, 'text')
        audio_grad, _ = model.getgrad(datainstance, 'audio')
        vision_grad, _ = model.getgrad(datainstance, 'vision')
    else:
        text_grad, _ = model.getgrad(datainstance, 'text', reverse=True)
        audio_grad, _ = model.getgrad(datainstance, 'audio', reverse=True)
        vision_grad, _ = model.getgrad(datainstance, 'vision', reverse=True)
    text_grad_norm = torch.norm(text_grad[0], p=1, dim=1)
    Y_vision, Y_audio = dataset.get_feature_def()
    Y_vision = [Y_vision[i] for i in vision_order]
    Y_audio = [Y_audio[i] for i in audio_order]
    words = model.getwords(datainstance)[:50]
    X = ['x'] * (50-len(words)) + words
    Z_vision = torch.absolute(vision_grad[0].T)
    Z_audio = torch.absolute(audio_grad[0].T)
    Z_vision_normed = torch.div(Z_vision, text_grad_norm).cpu().numpy()[vision_order]
    Z_audio_normed = torch.div(Z_audio, text_grad_norm).cpu().numpy()[audio_order]
    Z_text_normed = torch.unsqueeze(text_grad_norm[50-len(words):], 0).cpu().numpy()
    x_axis = [i for i in range(len(X))] 
    y_axis_vision = [0.8*i for i in range(len(Y_vision))]
    y_axis_audio = [i for i in range(len(Y_audio))]
    y_axis_text = [0, 1]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(x_axis[:len(words)]+[len(words)], y_axis_text, Z_text_normed, shading='auto', vmin=Z_text_normed.min(), vmax=Z_text_normed.max())
    plt.xticks(x_axis[:len(words)]+[len(words)], [' ']+words, rotation=60)
    plt.yticks(y_axis_text, [' ', 'feature gradient'])
    fname = 'mosei_grad_text_'+ str(idx) + '.png' if not reverse else 'mosei_correct_grad_text_'+ str(idx) + '.png'
    fig.savefig(savedir + fname)
    print(savedir + fname)

    
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(x_axis, y_axis_vision, Z_vision_normed, shading='nearest', vmin=Z_vision_normed.min(), vmax=Z_vision_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_vision, Y_vision)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), fau_colors):
        ticklabel.set_color(tickcolor)
    fname = 'mosei_grad_vision_'+ str(idx) + '.png' if not reverse else 'mosei_correct_grad_vision_'+ str(idx) + '.png'
    fig.savefig(savedir + fname)
    print(savedir + fname)

    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.pcolormesh(x_axis, y_axis_audio, Z_audio_normed, shading='nearest', vmin=Z_audio_normed.min(), vmax=Z_audio_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_audio, Y_audio)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), audio_colors):
        ticklabel.set_color(tickcolor)
    fname = 'mosei_grad_audio_'+ str(idx) + '.png' if not reverse else 'mosei_correct_grad_audio_'+ str(idx) + '.png'
    fig.savefig(savedir + fname)
    print(savedir + fname)

    

def visualize_grad_sparse(dataset, model, idx, feat, backward=None, reverse=False):
    #savedir = 'visuals/mosei_grad/mosei_'+str(idx)
    savedir = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if backward != None:
        savedir = 'private_test_scripts/mosei_simexp/mosei'+str(backward)+'/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    datainstance = dataset.getdata(idx)
    text_grad, _ = model.getgrad(datainstance, 'text', feat, prelinear=True)
    audio_grad, _ = model.getgrad(datainstance, 'audio', feat, prelinear=True)
    vision_grad, _ = model.getgrad(datainstance, 'vision', feat, prelinear=True)
    text_grad_norm = torch.norm(text_grad[0], p=1, dim=1)
    Y_vision, Y_audio = dataset.get_feature_def()
    Y_vision = [Y_vision[i] for i in vision_order]
    Y_audio = [Y_audio[i] for i in audio_order]
    words = model.getwords(datainstance)[:50]
    X = ['x'] * (50-len(words)) + words
    Z_vision = torch.absolute(vision_grad[0].T)
    Z_audio = torch.absolute(audio_grad[0].T)
    Z_vision_normed = torch.div(Z_vision, text_grad_norm).cpu().numpy()[vision_order]
    Z_audio_normed = torch.div(Z_audio, text_grad_norm).cpu().numpy()[audio_order]
    Z_text_normed = torch.unsqueeze(text_grad_norm[50-len(words):], 0).cpu().numpy()
    x_axis = [i for i in range(len(X))] 
    y_axis_vision = [0.8*i for i in range(len(Y_vision))]
    y_axis_audio = [i for i in range(len(Y_audio))]
    y_axis_text = [0, 1]
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(x_axis[:len(words)]+[len(words)], y_axis_text, Z_text_normed, shading='auto', vmin=Z_text_normed.min(), vmax=Z_text_normed.max())
    plt.xticks(x_axis[:len(words)]+[len(words)], [' ']+words, rotation=60)
    plt.yticks(y_axis_text, [' ', 'feature gradient'])
    data_id = idx if backward == None else backward
    fname1 = 'mosei_grad_text_' if not reverse else 'mosei_correct_grad_text_'
    fname2 = str(data_id) + '_feat_' + str(feat)
    fname3 = '' if backward == None else '_sample_' + str(idx)
    fname4 = '.png'
    fname = fname1 + fname2 + fname3 + fname4
    fig.savefig(savedir + fname)
    print(savedir + fname)

    
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.pcolormesh(x_axis, y_axis_vision, Z_vision_normed, shading='nearest', vmin=Z_vision_normed.min(), vmax=Z_vision_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_vision, Y_vision)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), fau_colors):
        ticklabel.set_color(tickcolor)
    data_id = idx if backward == None else backward
    fname1 = 'mosei_grad_vision_' if not reverse else 'mosei_correct_grad_vision_'
    fname2 = str(data_id) + '_feat_' + str(feat)
    fname3 = '' if backward == None else '_sample_' + str(idx)
    fname4 = '.png'
    fname = fname1 + fname2 + fname3 + fname4
    fig.savefig(savedir + fname)
    print(savedir + fname)

    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.pcolormesh(x_axis, y_axis_audio, Z_audio_normed, shading='nearest', vmin=Z_audio_normed.min(), vmax=Z_audio_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_audio, Y_audio)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), audio_colors):
        ticklabel.set_color(tickcolor)
    data_id = idx if backward == None else backward
    fname1 = 'mosei_grad_audio_' if not reverse else 'mosei_correct_grad_audio_'
    fname2 = str(data_id) + '_feat_' + str(feat)
    fname3 = '' if backward == None else '_sample_' + str(idx)
    fname4 = '.png'
    fname = fname1 + fname2 + fname3 + fname4
    fig.savefig(savedir + fname)
    print(savedir + fname)


def visualize_grad_double(dataset, model, idx, target_idxs, reverse=False):
    #savedir = 'visuals/mosei_grad/mosei_'+str(idx)
    savedir = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    datainstance = dataset.getdata(idx)
    text_grad, _ = model.getgrad(datainstance, 'text')
    audio_grad = model.getdoublegrad(datainstance, 'audio', target_idxs)
    vision_grad = model.getdoublegrad(datainstance, 'vision', target_idxs)
    text_grad_norm = torch.norm(text_grad[0], p=1, dim=1)
    Y_vision, Y_audio = dataset.get_feature_def()
    Y_vision = [Y_vision[i] for i in vision_order]
    Y_audio = [Y_audio[i] for i in audio_order]
    words = model.getwords(datainstance)[:50]
    target_words = [words[k] for k in target_idxs] 
    X = ['x'] * (50-len(words)) + words
    Z_vision = torch.absolute(vision_grad[0].T)
    Z_audio = torch.absolute(audio_grad[0].T)
    Z_vision_normed = torch.div(Z_vision, text_grad_norm).cpu().numpy()[vision_order]

    sums = np.sum(Z_vision_normed, axis=0)
    sorted_sums = np.argsort(sums)

    Z_audio_normed = torch.div(Z_audio, text_grad_norm).cpu().numpy()[audio_order]
    x_axis = [i for i in range(len(X))] 
    y_axis_vision = [0.8*i for i in range(len(Y_vision))]
    y_axis_audio = [i for i in range(len(Y_audio))]

    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title('SOG of vision w.r.t to words [' + ', '.join(target_words) + ']')
    ax.pcolormesh(x_axis, y_axis_vision, Z_vision_normed, shading='nearest', vmin=Z_vision_normed.min(), vmax=Z_vision_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_vision, Y_vision)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), fau_colors):
        ticklabel.set_color(tickcolor)
    fname = 'mosei_doublegrad_vision_'+ str(idx) + '.png' if not reverse else 'mosei_correct_doublegrad_vision_'+ str(idx) + '.png'    
    fig.savefig(savedir + fname)
    print(savedir + fname)

    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title('SOG of audio w.r.t to words [' + ', '.join(target_words) + ']')
    ax.pcolormesh(x_axis, y_axis_audio, Z_audio_normed, shading='nearest', vmin=Z_audio_normed.min(), vmax=Z_audio_normed.max())
    plt.xticks(x_axis, X, rotation=70)
    plt.yticks(y_axis_audio, Y_audio)
    for ticklabel, tickcolor in zip(plt.gca().get_yticklabels(), audio_colors):
        ticklabel.set_color(tickcolor)
    fname = 'mosei_doublegrad_audio_'+ str(idx) + '.png' if not reverse else 'mosei_correct_doublegrad_audio_'+ str(idx) + '.png'    
    fig.savefig(savedir + fname)
    print(savedir + fname)       


def analyzepointandvisualizeallgrad(
    params, dataidx, dataset, analysismodel, label, pathnum=95, k=5
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    print(topk)
    for i in topk:
        print(glmres["weight"][label][i])
    for j in range(k):
        visualize_grad_sparse(dataset, analysismodel, dataidx, topk[j])

def get_topk_feats_and_weights(
    params, dataidx, dataset, analysismodel, label, pathnum=95, k=5
):
    glmres = params["path"][pathnum]
    topk = glmres["weight"][label].squeeze().numpy().argsort()[-k:][::-1]
    weights = [glmres["weight"][label][i] for i in topk]
    return topk, weights

def get_sparse_info(dataset, model, params, pathnum=95, k=5):
    res = []
    for idx in range(dataset.length()):
        datainstance = dataset.getdata(idx)
        name = datainstance[4]
        resobj = model.forward(datainstance)
        prelinear = model.getprelinear(resobj)
        label = model.getpredlabel(resobj)
        glmres = params["path"][pathnum]
        allfeats = [i for i in range(180)]
        info = []
        for i in allfeats:
            weight = glmres["weight"][label][i].item()
            info.append((i, weight, prelinear[i].item()))
        res.append((idx, name, info))

    return res

def get_sparse_info_correct(dataset, model, params, pathnum=95, k=5):
    res = []
    for idx in range(dataset.length()):
        datainstance = dataset.getdata(idx)
        name = datainstance[4]
        resobj = model.forward(datainstance)
        prelinear = model.getprelinear(resobj)
        label = model.getcorrectlabel(datainstance)
        glmres = params["path"][pathnum]
        allfeats = [i for i in range(180)]
        info = []
        for i in allfeats:
            weight = glmres["weight"][label][i].item()
            info.append((i, weight, prelinear[i].item()))
        res.append((idx, name, info))

    return res    


        

if __name__=='__main__':

    params = torch.load('ckpt/moseisparselinearmodel.pt') 
    dataset = MOSEIDataset()
    model = MOSEIMULT()
    #sparse_info = get_sparse_info(dataset, model, params) 
    #print(sparse_info[3])

    '''
    #idx = 13
    #datainstance = dataset.getdata(idx)
    #resobj = model.forward(datainstance)
    #pred = model.getpredlabel(resobj)
    for i in [j*10 for j in range(1)]:
        datainstance = dataset.getdata(i)
        resobj = model.forward(datainstance)
        pred = model.getpredlabel(resobj)
        #visualize_grad(model, i)
        analyzepointandvisualizeallgrad(
            params, i, model, pred, pathnum=95, k=5)'''  
    targets = [[12,13,14,16,18,19,20], [4,5,6,7,8,10,11], [19,20,21,22], [3,4,5,6,7], [12,14,15,16,17], 
               [9,11,13,14,15], [1,3,4,6,7,8], [2,3,4,5,6,7,8], [2,3,4,5], [0,2,3,4], [2,3,4],
               [1,2,3,4,5,6], [3,4,5,6,7], [0,1,2], [8,9,10,11,12,14,15], [2,3,4,5], [2,4,5,7], 
               [16,18,19,20,21], [2,3,4,6], [4,5,6,7]]
    for i in range(11,12):
        if not os.path.isdir('private_test_scripts/mosei_doublegrad/mosei'+str(i)):
            os.mkdir('private_test_scripts/mosei_doublegrad/mosei'+str(i))
        visualize_grad_double(dataset, model, i, targets[i])            