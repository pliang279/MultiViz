import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.insert(1,os.getcwd())
sys.path.insert(2,'/home/paul/yiwei/multimodal_analysis/structured-framework/models/mult')
from models.analysismodel import analysismodel
import copy

from datasets.mosei2 import*


class MOSEIMULT(analysismodel):
    def __init__(self ,device='cuda'):
        self.model = torch.load('models/mult/mosei_sentimult_MULT_a_0.3_e_0.2_o_0.3_res_0.1.pt').to(device)
        self.modalitynames = ['text','audio','vision']
        self.modalitytypes = ['timeseries','timeseries','timeseries']
        self.answermapping = ['positive', 'negative']
        self.device=device

    def getunimodaldata(self,datainstance,modality):
        return datainstance[self.modalitynames.index(modality)]

    def getcorrectlabel(self,datainstance):
        if datainstance[3] >= 0:
            return 0
        return 1

    def getwords(self, datainstance):
        return datainstance[5]    

    def forward(self,datainstance):
        t, a, v = datainstance[:3]
        tb = torch.unsqueeze(t,0).to(self.device)
        ab = torch.unsqueeze(a,0).to(self.device)
        vb = torch.unsqueeze(v,0).to(self.device)
        logit, last_hs = self.model(tb, ab, vb)
        binary_logit = torch.tensor([logit[0][0], -logit[0][0]]).to(self.device)
        return logit[0].detach(), last_hs[0].detach(), binary_logit

    def forwardbatch(self,datainstances):
        outs=[]
        for di in datainstances:
            outs.append(self.forward(di))
        #texts = [datainstance[0] for datainstance in datainstances]
        #audios = [datainstance[0] for datainstance in datainstances]
        #visions = [datainstance[0] for datainstance in datainstances]
        #tb = torch.stack(texts)
        #ab = torch.stack(audios)
        #vb = torch.stack(visions)
        #logit, last_hs = self.model(tb, ab, vb)
        return outs

    def getlogitsize(self):
        return 2

    def getlogit(self,resultobj):
        return resultobj[2] 

    def getprelinear(self,resultobj):
        return resultobj[1]

    def getpredlabel(self,resultobj):
        if resultobj[0].item() >= 0:
            return 0
        return 1

    def getpredanswer(self,resultobj):
        label = self.getpredlabel(resultobj)
        return self.answermapping[label]    

    def getprelinearsize(self):
        return 180

    def replaceunimodaldata(self,datainstance,modality,newdata):
        c = copy.deepcopy(datainstance)
        c[self.modalitynames.index(modality)] = newdata
        return c

    def getgrad(self, datainstance, modality, feat=None, prelinear=False, alltarget=False, reverse=False):
        if prelinear:
            assert(feat != None)
            self.model.zero_grad()
            t, a, v = datainstance[:3]
            tb = torch.unsqueeze(t,0).to(self.device)
            ab = torch.unsqueeze(a,0).to(self.device)
            vb = torch.unsqueeze(v,0).to(self.device)
            feats_list = [tb, ab, vb]
            modality_with_grad = feats_list[self.modalitynames.index(modality)]
            modality_with_grad.requires_grad = True
            _, last_hs, = self.model(tb, ab, vb) 
            last_hs[0][feat].backward()
            grad = modality_with_grad.grad.detach()

            return grad, feats_list

        self.model.zero_grad()
        t, a, v = datainstance[:3]
        tb = torch.unsqueeze(t,0).to(self.device)
        ab = torch.unsqueeze(a,0).to(self.device)
        vb = torch.unsqueeze(v,0).to(self.device)
        feats_list = [tb, ab, vb]
        modality_with_grad = feats_list[self.modalitynames.index(modality)]
        #modality_with_grad.requires_grad = True   
        for modality in feats_list:
            modality.requires_grad = True    
        logit, _, = self.model(tb, ab, vb)
        if alltarget:
            logit.backward(create_graph=True)
        elif reverse:
            (-logit).backward()    
        else:
            logit.backward()    
        grad = modality_with_grad.grad

        return grad, feats_list

    def getdoublegrad(self, datainstance, modality, targetwords):
        res, feats_list = self.getgrad(
            datainstance, 'text', alltarget=True
        )
        ac = torch.zeros(res[0][0].shape)
        for id in targetwords:
            ac += torch.abs(res[0][id]) 
        ac = torch.sum(ac)
        feat = feats_list[2] if modality == 'image' else feats_list[1]
        rets = torch.autograd.grad(ac, feat)
        return rets[0]   




if __name__=='__main__':
    dataset = MOSEIDataset()
    model = MOSEIMULT()

    idx = 13
    datainstance = dataset.getdata(idx)
    resobj = model.forward(datainstance)
    g = model.getgrad(datainstance, 'text', 10, True)
    gg = model.getdoublegrad(datainstance, 'audio', [30, 31])
    print(gg.shape)

    ###############################
    # Sparse Linear Model         #
    ###############################
    '''
    from analysis.SparseLinearEncoding import*

    val_dataset = MOSEIDataset_orig()
    train_dataset = MOSEIDataset_orig('train')
    test_dataset = MOSEIDataset_orig('test')

    val_instances = [val_dataset.getdata(i) for i in tqdm(range(val_dataset.length()))]
    train_instances = [train_dataset.getdata(i) for i in tqdm(range(train_dataset.length()))]
    test_instances = [test_dataset.getdata(i) for i in tqdm(range(test_dataset.length()))]

    val_embeds = getembeds(val_instances, model)
    train_embeds = getembeds(train_instances, model)
    test_embeds = getembeds(test_instances, model)

    params, (accuracies,sparse_cnts) = get_sparse_linear_model(model, train_embeds, val_embeds, 
                                                               test_embeds, modelsavedir='ckpt/moseisparselinearmodel.pt')
    '''                                           
    