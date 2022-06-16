import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from matplotlib import ticker

def normalize255(t, fac=255.0):
    upmost = torch.max(torch.abs(t))
    return (fac * t / upmost).long()


def heatmap2d(t, savename, orig=None,fivebyfive=True):
    plt.clf()
    pxs = torch.zeros(len(t), len(t[0]), 3)
    t = torch.clamp(t, -255, 255)
    if fivebyfive:
        t = fivebyfivefn(t)
    for i in range(len(t)):
        for j in range(len(t[0])):
            if t[i, j] >= 0:
                pxs[i, j, 1] = t[i, j]
            else:
                pxs[i, j, 0] = 0 - 1 - t[i, j]
    img = Image.fromarray(pxs.long().numpy().astype(np.uint8))
    if orig is not None:
        img2 = cv2.resize(np.asarray(Image.open(orig)), (pxs.shape[1], pxs.shape[0]))
        plt.imshow(img2)
        plt.imshow(img, alpha=0.4)
    else:
        plt.imshow(img)
    plt.savefig(savename)

def fivebyfivefn(mx):
    maxx = torch.zeros(len(mx),len(mx[0]))
    for i in range(len(maxx)):
        for j in range(len(maxx[0])):
            maxval = 0
            for iii in range(5):
                for jjj in range(5):
                    ii = i +iii - 2
                    jj = j +jjj - 2
                    if ii >= 0 and ii < len(maxx) and jj >= 0 and jj < len(maxx[0]) and  mx[ii][jj] > maxval:
                        maxval = mx[ii][jj]
            maxx[i][j] = maxval
    return maxx
                    


def textmap(words, weights, savedir, tops=10):
    c = torch.argsort(torch.abs(weights))
    if len(c) > tops:
        c = c[-tops:]
    c = c.cpu().tolist()
    plt.clf()
    vals = []
    names = []
    for i in c:
        vals.append(weights[i].item())
        names.append(words[i])
    colors = ["green" if x >= 0 else "red" for x in vals]
    pos = np.arange(len(colors)) + 0.5
    plt.barh(pos, vals, align="center", color=colors)
    plt.yticks(pos, names)
    plt.savefig(savedir,bbox_inches='tight')

def heatmapts(xwords,ywords,grads,savedir):
    plt.clf()
    #print(grads)
    cm = plt.get_cmap('PiYG')
    fig,ax = plt.subplots()
    cax = ax.imshow(grads,cmap=cm)
    fig.colorbar(cax).set_label('first order gradient',rotation=270)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    xwords.insert(0,'none')
    ywords.insert(0,'none')
    ax.set_xticklabels(xwords)
    ax.set_yticklabels(ywords)
    plt.savefig(savedir,bbox_inches='tight')
