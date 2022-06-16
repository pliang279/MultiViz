import json
import sys
import os
sys.path.insert(1,os.getcwd())

from visualizations.visualizevideo import*
from datasets.clevr import*
from models.clevr_cnnlstmsa import*
from visualizations.visualizegradient import*


dataset = CLEVRDataset()
model = CLEVRCNNLSTMSA()
targets = {}

def generate_json(idx):
    datainstance = dataset.getdata(idx)
    correct_label = model.getcorrectlabel(datainstance)
    correct_answer = model.getcorrectanswer(datainstance)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)
    pred_answer = model.getpredanswer(resobj)
    
    #topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, pred_label)


    info = dict()
    info['metadata'] = {
        'dataset': 'CLEVR',
        'split': 'val',
        'id': idx,
        'labels': dict()
    }
    info['metadata']['labels'][str(pred_label)] = pred_answer
    info['metadata']['labels'][str(correct_label)] = correct_answer

    image, text = datainstance[0], datainstance[1]
    info['instance'] = {
        'image': image,
        'text': text,
        'correct-answer': correct_answer,   
        'correct-answer-id': correct_label,
        'pred-answer': pred_answer,
        'pred-id': pred_label
    }
    info['labels'] = dict()

    #pred_label
    info['labels'][str(pred_label)] = dict()
    info['labels'][str(pred_label)]['classname'] = pred_label
    info['labels'][str(pred_label)]['overviews'] = dict()
    info['labels'][str(pred_label)]['overviews']['UnimodalGradient'] = {
        'description': 'Gradient-based explanation on the text and image features ran directly on the output logit',
        'text': 'clevrtrash_grad_text_'+str(idx)+'.png', 
        'image': 'clevrtrash_grad_image_'+str(idx)+'.png', 
    }

    target_idxs = targets[idx]
    words = model.parse(text)
    target_words = [words[k] for k in target_idxs]

    info['labels'][str(pred_label)]['overviews']['SecondOrderGradient'] = {
        'description': 'Gradient-based explanation on the text and image features ran directly on the output logit',
        'words': target_words,
        'text': 'clevrtrash_sog_text_'+str(idx)+'.png', 
        'image': 'clevrtrash_sog_image_'+str(idx)+'.png', 
    }

    return info


def generate_data(idx):
    data_instance = dataset.getdata(idx)
    correct_label = model.getcorrectlabel(datainstance)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)

    imgfile, question = datainstance[0], datainstance[1]
    img = Image.open(imgfile)
    outdir = 'private_test_scripts/clevrtrash_simexp/'

    # Unimodal Gradient
    _, grad, _ = model.getgrad(data_instance, pred_label, prelinear=True)
    res, parsed, _, _ = model.getgradtext(data_instance, pred_label, prelinear=True)
    grads = torch.sum(torch.abs(grad).squeeze(), dim=0)
    t = normalize255(grads)
    heatmap2d(t, outdir + 'clevrtrash_grad_image_' + str(idx) + '.png', imgfile)  
    textmap(parsed, torch.abs(res), outdir + 'clevrtrash_grad_text_' + str(idx) + '.png')

    # Second Order Gradient
    target_idxs = targets[idx]
    words = model.parse(text)
    target_words = [words[k] for k in target_idxs]
    dgrad, _, _ = model.getdoublegrad(data_instance, pred_label, target_words)
    
    )
