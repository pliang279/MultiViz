from glm_saga.elasticnet import glm_saga,get_device
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from tqdm import tqdm
#from analysis.utils import accuracy


def getembeds(datainstances,analysismodel,reallabel=True):
    embeds=[]
    ind = 0
    for resultobj in analysismodel.forwardbatch(datainstances):
        if reallabel:
            embeds.append((analysismodel.getprelinear(resultobj),analysismodel.getcorrectlabel(datainstances[ind]),ind))
        else:
            embeds.append((analysismodel.getprelinear(resultobj),analysismodel.getpredlabel(resultobj),ind))
        ind += 1
    return embeds

def get_sparse_linear_model(analysismodel,trainembeds,valembeds,testembeds,modelsavedir='ckpt/sparselinearmodel.pt',model_idxs=np.arange(100)):
    linear = torch.nn.Linear(analysismodel.getprelinearsize(),analysismodel.getlogitsize()).to(analysismodel.device)
    torch.nn.init.constant_(linear.weight, 0)
    torch.nn.init.constant_(linear.bias, 0)
    training_loader = DataLoader(trainembeds,shuffle=True,batch_size=64)
    val_loader = DataLoader(valembeds,shuffle=False,batch_size=64)
    test_loader = DataLoader(testembeds,shuffle=False,batch_size=64)
    
    params = glm_saga(linear, 
                training_loader, 
                0.1, 
                500, 
                0.99, 
                val_loader=val_loader,
                test_loader=test_loader,
                n_classes=analysismodel.getlogitsize(),
                n_ex=len(trainembeds), 
                checkpoint='./ckpt/',
                verbose=50, 
                tol=1e-4, 
                lookbehind=3, 
                lr_decay_factor=1,
                group=False, 
                epsilon=0.001,
                preprocess=Identity(linear))
    torch.save(params,modelsavedir)
    accuracies = []
    sparse_cnts = []
    for model_idx in tqdm(model_idxs):
        pretrained_weights = params['path'][model_idx]['weight']
        pretrained_bias = params['path'][model_idx]['bias']
        linear.load_state_dict({'weight': pretrained_weights, 'bias': pretrained_bias})
        linear.eval()
        accuracy, sparse_cnt, numel = evaluate_sparse_linear_model(test_loader, linear, analysismodel.device)
        # sparsity = sparse_cnt/numel
        sparse_cnts.append(sparse_cnt.cpu().item())
        accuracies.append(accuracy)  
    return params, (accuracies,sparse_cnts)

def getresonly(params,analysismodel,testembeds,model_idxs=np.arange(100)):

    test_loader = DataLoader(testembeds,shuffle=False,batch_size=64)
    linear = torch.nn.Linear(analysismodel.getprelinearsize(),analysismodel.getlogitsize()).to(analysismodel.device)
    accuracies = []
    sparse_cnts = []
    for model_idx in tqdm(model_idxs):
        pretrained_weights = params['path'][model_idx]['weight']
        pretrained_bias = params['path'][model_idx]['bias']
        linear.load_state_dict({'weight': pretrained_weights, 'bias': pretrained_bias})
        linear.eval()
        accuracy, sparse_cnt, numel = evaluate_sparse_linear_model(test_loader, linear, analysismodel.device)
        # sparsity = sparse_cnt/numel
        sparse_cnts.append(sparse_cnt.cpu().item())
        accuracies.append(accuracy)  
    return params, (accuracies,sparse_cnts)


def evaluate_sparse_linear_model(test_loader, model,device):

    model.eval()
    preds = []
    labels = []
    softmax = torch.nn.Softmax(dim=1)

    for embed, label, index in test_loader:
        embed = embed.to(device)
        with torch.no_grad():
            logits = model(embed)
        batch_preds = softmax(logits).argsort(descending=True, dim=1)[:, 0]
        preds.append(batch_preds.cpu())
        labels.append(label)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    numel = model.weight.data.numel()
    sparse_cnt = (model.weight.data.abs() < 1e-5).sum()
    acc = (preds == labels).float().mean()

    return acc.item(), sparse_cnt, numel

class Identity():
    def __init__(self,a):
        #super(Identity,self).__init__()
        self.device=get_device(a)
    def __call__(self,b):
        return b

    

