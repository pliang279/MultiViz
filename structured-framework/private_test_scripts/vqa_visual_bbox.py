import os
import sys
sys.path.insert(1,os.getcwd())
from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

datas = VQADataset('val')
analysismodel = VQALXMERT(device='cuda:1')

idx = 255
datainstance = datas.getdata(idx)
image = analysismodel.getunimodaldata(datainstance, 'image')
question = analysismodel.getunimodaldata(datainstance, 'text')
resobj = analysismodel.forward(datainstance)
pred_idx = analysismodel.getpredlabel(resobj)
pred = datas.classnames()[pred_idx]
feat_grad, bboxs = analysismodel.getfeatgrad(datainstance, pred_idx)

grad_norms = torch.linalg.norm(feat_grad, ord=1, dim=1)
_, max_norm_idxs = torch.topk(grad_norms, 5)
#normed_bbox = bboxs[max_norm_idxs]
#bbox = (normed_bbox * 224).to(torch.int32).cpu().numpy()

im = Image.fromarray(image)
fig, ax = plt.subplots()
ax.imshow(im)
for i in max_norm_idxs:
    color = 'r'
    if i == max_norm_idxs[0]:
        color = 'g'
    b = bboxs[i]
    bb = (b * 224).cpu().to(torch.int32).numpy()
    rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
#rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
#ax.add_patch(rect)
ax.set_title('Q:'+question+'\n'+'A:'+pred)
plt.savefig('visuals/grad_bbox/vqa_grad_bbox_'+str(idx)+'.png')

