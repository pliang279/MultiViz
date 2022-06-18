import json
import sys
import os

from visualizations.visualizevideo import*
from visualizations.visualizemosei import*

dataset = MOSEIDataset()
model = MOSEIMULT()
params = torch.load('ckpt/moseisparselinearmodel.pt') 
sparse_info = get_sparse_info(dataset, model, params)
correct_sparse_info = get_sparse_info_correct(dataset, model, params)

def generate_json(idx):
    datainstance = dataset.getdata(idx)
    video_name = datainstance[4] + '.mp4'
    correct_label = dataset.get_correct_label(datainstance)
    correct_answer = dataset.get_correct_answer(datainstance)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)
    pred_answer = model.getpredanswer(resobj)


    info = dict()
    info['metadata'] = {
        'dataset': 'MOSEI',
        'split': 'val',
        'id': idx,
        'labels': dict()
    }
    info['metadata']['labels'][str(pred_label)] = pred_answer
    info['metadata']['labels'][str(correct_label)] = correct_answer

    script = get_script(dataset, model, idx)
    info['instance'] = {
        'video': video_name,
        'script': script,
        'correct-answer': correct_answer,   
        'correct-answer-id': correct_label,
        'pred-answer': pred_answer,
        'pred-id': pred_label
    }
    info['labels'] = dict()

    #pred_label
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, pred_label)
    info['labels'][str(pred_label)] = dict()
    info['labels'][str(pred_label)]['overviews'] = dict()
    info['labels'][str(pred_label)]['overviews']['gradient'] = {
        'description': 'Gradient-based explanation on the text, vision and audio features ran directly on the output logit',
        'video': 'mosei_grad_'+str(idx)+'.mp4',
        'text': 'mosei_grad_text_'+str(idx)+'.png', 
        'vision': 'mosei_grad_vision_'+str(idx)+'.png', 
        'audio': 'mosei_grad_audio_'+str(idx)+'.png'
    }

    target_idxs = targets[idx]
    words = model.getwords(datainstance)[:50]
    target_words = [words[k] for k in target_idxs] 
    sog = dict()
    sog['description'] = "Second order gradient explanation on vision and text features ran on the first order gradients of the target words"
    sog['words'] = words
    sog['target-words-id'] = target_idxs
    sog['target-words'] = target_words
    sog['vision'] = 'mosei_doublegrad_vision_'+ str(idx) + '.png'
    sog['audio'] = 'mosei_doublegrad_audio_'+ str(idx) + '.png'
    info['labels'][str(pred_label)]['overviews']['SOG'] = sog

    info['labels'][str(pred_label)]['features'] = []
    for i in range(len(topk_feats)):
        d = dict()
        feat = topk_feats[i].item()
        weight = topk_weights[i].item()
        d['id'] = feat
        d['weight'] = weight
        d['forward'] = {
            'video': 'mosei_grad_'+str(idx)+'_feat_'+str(feat)+'.mp4',
            'text': 'mosei_grad_text_'+str(idx)+'_feat_'+str(feat)+'.png', 
            'vision': 'mosei_grad_vision_'+str(idx)+'_feat_'+str(feat)+'.png', 
            'audio': 'mosei_grad_audio_'+str(idx)+'_feat_'+str(feat)+'.png'
        }
        d['forward-descriptions'] =  "Gradient-based explanation on the text, vision and audio features with respect to the value of this feature neuron"
        d['backward'] = []
        sorted_sparse_info = sorted(sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            orig_script = get_script(dataset, model, instance_id)
            instance_d = {
                'id': instance_id, 
                'orig_video': instance_name+'.mp4',
                'orig_script': orig_script,
                'video': 'mosei_grad_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.mp4', 
                'text': 'mosei_grad_text_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png', 
                'vision': 'mosei_grad_vision_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png', 
                'audio': 'mosei_grad_audio_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png'
            }
            d['backward'].append(instance_d)
        d['backward-descriptions'] = "Three examples selected from the validation set that activates this feature neuron the most, along with gradient-based explanation with respect to the value of this feature neuron in each example" 
        info['labels'][str(pred_label)]['features'].append(d)

    #correct_label
    if correct_label == pred_label:
        return info
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, correct_label)
    info['labels'][str(correct_label)] = dict()
    info['labels'][str(correct_label)]['overviews'] = dict()
    info['labels'][str(correct_label)]['overviews']['gradient'] = {
        'description': 'Gradient-based explanation on the text, vision and audio features ran directly on the output logit',
        'video': 'mosei_correct_grad_'+str(idx)+'.mp4',
        'text': 'mosei_correct_grad_text_'+str(idx)+'.png', 
        'vision': 'mosei_correct_grad_vision_'+str(idx)+'.png', 
        'audio': 'mosei_correct_grad_audio_'+str(idx)+'.png'
    }

    sog = dict()
    sog['description'] = "Second order gradient explanation on vision and text features ran on the first order gradients of the target words"
    sog['words'] = words
    sog['target-words-id'] = target_idxs
    sog['target-words'] = target_words
    sog['vision'] = 'mosei_correct_doublegrad_vision_'+ str(idx) + '.png'
    sog['audio'] = 'mosei_correct_doublegrad_audio_'+ str(idx) + '.png'
    info['labels'][str(correct_label)]['overviews']['SOG'] = sog

    info['labels'][str(correct_label)]['features'] = []
    for i in range(len(topk_feats)):
        d = dict()
        feat = topk_feats[i].item()
        weight = topk_weights[i].item()
        d['id'] = feat
        d['weight'] = weight
        d['forward'] = {
            'video': 'mosei_correct_grad_'+str(idx)+'_feat_'+str(feat)+'.mp4',
            'text': 'mosei_correct_grad_text_'+str(idx)+'_feat_'+str(feat)+'.png', 
            'vision': 'mosei_correct_grad_vision_'+str(idx)+'_feat_'+str(feat)+'.png', 
            'audio': 'mosei_correct_grad_audio_'+str(idx)+'_feat_'+str(feat)+'.png'
        }
        d['forward-descriptions'] =  "Gradient-based explanation on the text, vision and audio features with respect to the value of this feature neuron"
        d['backward'] = []
        sorted_sparse_info = sorted(correct_sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            orig_script = get_script(dataset, model, instance_id)
            instance_d = {
                'id': instance_id, 
                'orig_video': instance_name+'.mp4',
                'orig_script': orig_script,
                'video': 'mosei_correct_grad_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.mp4', 
                'text': 'mosei_correct_grad_text_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png', 
                'vision': 'mosei_correct_grad_vision_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png', 
                'audio': 'mosei_correct_grad_audio_'+str(idx)+'_feat_'+str(feat)+'_sample_'+str(instance_id)+'.png'
            }
            d['backward'].append(instance_d)
        d['backward-descriptions'] = "Three examples selected from the validation set that activates this feature neuron the most, along with gradient-based explanation with respect to the value of this feature neuron in each example" 
        info['labels'][str(correct_label)]['features'].append(d)      

    return info


def generate_video_data(idx):
    datainstance = dataset.getdata(idx)
    video_name = datainstance[4] + '.mp4'
    correct_label = dataset.get_correct_label(datainstance)
    #correct_answer = dataset.get_correct_answer(datainstance)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)
    #pred_answer = model.getpredanswer(resobj)

    dirname = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if not os.path.exists(dirname + video_name):
        #subprocess.check_call('cp data/Raw/Videos/'+video_name+' '+dirname, shell=True)
        clip_data(dataset, model, idx)

    # pred label
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, pred_label)
    process_data(dataset, model, idx, reverse=False)    

    for i in range(len(topk_feats)):
        feat = topk_feats[i].item()
        #weight = topk_weights[i].item()
        process_data(dataset, model, idx, feat=feat, reverse=False)

        sorted_sparse_info = sorted(sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            if not os.path.exists(dirname + instance_name):
                clip_data(dataset, model, instance_id, idx)
            process_data(dataset, model, instance_id, feat=feat, backward=idx, reverse=False)

    # correct label
    if correct_label == pred_label:
        return
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, correct_label)
    process_data(dataset, model, idx, reverse=True)    

    for i in range(len(topk_feats)):
        feat = topk_feats[i].item()
        #weight = topk_weights[i].item()
        process_data(dataset, model, idx, feat=feat, reverse=True)

        sorted_sparse_info = sorted(sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            if not os.path.exists(dirname + instance_name):
                clip_data(dataset, model, instance_id, idx)
            process_data(dataset, model, instance_id, feat=feat, backward=idx, reverse=True)
        

def generate_heatmap_data(idx):
    datainstance = dataset.getdata(idx)
    #video_name = datainstance[4] + '.mp4'
    correct_label = dataset.get_correct_label(datainstance)
    #correct_answer = dataset.get_correct_answer(datainstance)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)
    #pred_answer = model.getpredanswer(resobj)
    target_idxs = targets[idx]

    # pred label
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, pred_label)
    visualize_grad(dataset, model, idx)
    visualize_grad_double(dataset, model, idx, target_idxs)
    for i in range(len(topk_feats)):
        feat = topk_feats[i].item()
        #weight = topk_weights[i].item()
        visualize_grad_sparse(dataset, model, idx, feat=feat)

        sorted_sparse_info = sorted(sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            visualize_grad_sparse(dataset, model, instance_id, feat=feat, backward=idx)

    # correct label
    topk_feats, topk_weights = get_topk_feats_and_weights(params, idx, dataset, model, correct_label)
    visualize_grad(dataset, model, idx, reverse=True)
    visualize_grad_double(dataset, model, idx, target_idxs, reverse=True)
    for i in range(len(topk_feats)):
        feat = topk_feats[i].item()
        #weight = topk_weights[i].item()
        visualize_grad_sparse(dataset, model, idx, feat=feat, reverse=True)

        sorted_sparse_info = sorted(sparse_info, key=lambda x: x[2][feat][2], reverse=True)
        backward_instances = [sorted_sparse_info[i][:2] for i in range(3)]
        for instance_id, instance_name in backward_instances:
            visualize_grad_sparse(dataset, model, instance_id, feat=feat, backward=idx, reverse=True)                    


