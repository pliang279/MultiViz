import torch
import numpy as np

def grad_to_saliency(sample, grad):
    saliency_grad = grad.detach() * sample.detach()
    saliency_grad = saliency_grad.cpu().numpy()
    norm = np.linalg.norm(saliency_grad.flatten(), ord=1)
    saliency_grad = torch.tensor([(-e) / norm for e in saliency_grad])
    return saliency_grad

def grad_to_saliency2(sample, grad):
    saliency_grad = grad.detach()
    saliency_grad = saliency_grad.cpu().numpy()
    norm = np.linalg.norm(saliency_grad.flatten(), ord=1)
    saliency_grad = torch.tensor([(-e) / norm for e in saliency_grad])
    return saliency_grad

def get_saliency_map(datainstance,analysismodel,target,multiplyorig=False):
    orig,grad = analysismodel.getgrad(datainstance,target)
    rets=[]
    for i in range(len(grad)):
        if multiplyorig:
            rets.append(grad_to_saliency(orig[i],grad[i]))
        else:
            rets.append(grad_to_saliency2(orig[i],grad[i]))
    return rets

