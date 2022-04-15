import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def normalize255(t):
    upmost = torch.max(torch.abs(t))
    return (255.0*t/upmost).long()



def heatmap2d(t,savename,orig=None):
    pxs=torch.zeros(len(t),len(t[0]),3)
    for i in range(len(t)):
        for j in range(len(t[0])):
            if t[i,j] >= 0:
                pxs[i,j,1]=t[i,j]
            else:
                pxs[i,j,0]=0-1-t[i,j]
    img = Image.fromarray(pxs.long().numpy().astype(np.uint8))
    if orig is not None:
        img2 = Image.open(orig)
        plt.imshow(img2)
        plt.imshow(img,alpha=0.7)
    else:
        plt.imshow(img)
    plt.savefig(savename)
