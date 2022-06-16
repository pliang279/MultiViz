import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.insert(1,os.getcwd())
sys.path.insert(2,'/home/paul/yiwei/multimodal_analysis/structured-framework/models/mult')
from models.analysismodel import analysismodel
import copy

from datasets.mosi import MOSIDataset


class MOSIMULT(analysismodel):
    def __init__(self ,device='cuda:0'):
        self.model = torch.load('models/mult/mosimult_MULT_a_0.1_e_0.4_o_0.5_res_0.1.pt').to(device)
        self.modalitynames = ['text','audio','vision']
        self.modalitytypes = ['timeseries','timeseries','timeseries']
        self.labelmapping = ['positive', 'negative']
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

    def getprelinearsize(self):
        return 180

    def replaceunimodaldata(self,datainstance,modality,newdata):
        c = copy.deepcopy(datainstance)
        c[self.modalitynames.index(modality)] = newdata
        return c

    def getgrad(self, datainstance, modality):
        self.model.zero_grad()
        t, a, v = datainstance[:3]
        tb = torch.unsqueeze(t,0).to(self.device)
        ab = torch.unsqueeze(a,0).to(self.device)
        vb = torch.unsqueeze(v,0).to(self.device)
        feats_list = [tb, ab, vb]
        modality_with_grad = feats_list[self.modalitynames.index(modality)]
        modality_with_grad.requires_grad = True       
        logit, _ = self.model(tb, ab, vb)
        logit.backward()
        grad = modality_with_grad.grad.detach()

        return grad




if __name__=='__main__':
    dataset = MOSIDataset()
    model = MOSIMULT()

    idx = 5
    datainstance = dataset.getdata(idx)
    resobj = model.forward(datainstance)
    pred_label = model.getpredlabel(resobj)
    classnames = dataset.classnames()

    ###############################
    # Gradient Visualization      #
    ###############################
    
    import matplotlib.pyplot as plt
    import numpy as np

    def visualize_grad(idx):
        savedir = 'visuals/mosei_grad/mosei_'+str(idx)
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        datainstance = dataset.getdata(idx)
        text_grad = model.getgrad(datainstance, 'text')
        audio_grad = model.getgrad(datainstance, 'audio')
        vision_grad = model.getgrad(datainstance, 'vision')
        Y_vision, Y_audio = dataset.get_feature_def()
        words = model.getwords(datainstance)
        X = ['x'] * (50-len(words)) + words if len(words) < 50 else words[:50]
        Z_vision = np.absolute(vision_grad[0].cpu().numpy().T)
        Z_audio = np.absolute(audio_grad[0].cpu().numpy().T)
        x_axis = [i for i in range(len(X))] 
        y_axis_vision = [0.8*i for i in range(len(Y_vision))]
        y_axis_audio = [i for i in range(len(Y_audio))]
        
        #text_embeds = model.getunimodaldata(datainstance, 'text')

        plt.clf()
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.pcolormesh(x_axis, y_axis_vision, Z_vision, shading='nearest', vmin=Z_vision.min(), vmax=Z_vision.max())
        plt.xticks(x_axis, X, rotation=70)
        plt.yticks(y_axis_vision, Y_vision)
        fig.savefig(savedir + '/mosei_grad_vision_'+ str(idx) + '.png')

        plt.clf()
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.pcolormesh(x_axis, y_axis_audio, Z_audio, shading='nearest', vmin=Z_vision.min(), vmax=Z_vision.max())
        plt.xticks(x_axis, X, rotation=70)
        plt.yticks(y_axis_audio, Y_audio)
        fig.savefig(savedir + '/mosei_grad_audio_'+ str(idx) + '.png')

        #Visualization for non-null words
        if len(words) > 0:
            Z_vision_2 = np.absolute(vision_grad[0][50-len(words):].cpu().numpy().T)
            Z_audio_2 = np.absolute(audio_grad[0][50-len(words):].cpu().numpy().T)
            x_axis_2 = [i for i in range(len(words))]

            plt.clf()
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.pcolormesh(x_axis_2, y_axis_vision, Z_vision_2, shading='nearest', vmin=Z_vision_2.min(), vmax=Z_vision_2.max())
            plt.xticks(x_axis_2, words, rotation=70)
            plt.yticks(y_axis_vision, Y_vision)
            fig.savefig(savedir + '/mosei_grad_vision_words_'+ str(idx) + '.png')

            plt.clf()
            fig, ax = plt.subplots(figsize=(16, 16))
            ax.pcolormesh(x_axis_2, y_axis_audio, Z_audio_2, shading='nearest', vmin=Z_vision_2.min(), vmax=Z_vision_2.max())
            plt.xticks(x_axis_2, words, rotation=70)
            plt.yticks(y_axis_audio, Y_audio)
            fig.savefig(savedir + '/mosei_grad_audio_words_'+ str(idx) + '.png')

        if len(words) < 50:
            Z_vision_3 = np.absolute(vision_grad[0][:50-len(words)].cpu().numpy().T)
            Z_audio_3 = np.absolute(audio_grad[0][:50-len(words)].cpu().numpy().T)
            x_axis_3 = [i for i in range(50-len(words))]

            plt.clf()
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.pcolormesh(x_axis_3, y_axis_vision, Z_vision_3, shading='nearest', vmin=Z_vision_3.min(), vmax=Z_vision_3.max())
            plt.xticks(x_axis_3, ['x']*(50-len(words)), rotation=70)
            plt.yticks(y_axis_vision, Y_vision)
            fig.savefig(savedir + '/mosei_grad_vision_null_'+ str(idx) + '.png')

            plt.clf()
            fig, ax = plt.subplots(figsize=(16, 16))
            ax.pcolormesh(x_axis_3, y_axis_audio, Z_audio_3, shading='nearest', vmin=Z_vision_3.min(), vmax=Z_vision_3.max())
            plt.xticks(x_axis_3, ['x']*(50-len(words)), rotation=70)
            plt.yticks(y_axis_audio, Y_audio)
            fig.savefig(savedir + '/mosei_grad_audio_null_'+ str(idx) + '.png')    

    visualize_grad(idx)  
    
    

    ###############################
    # Sparse Linear Model         #
    ###############################
    '''
    from analysis.SparseLinearEncoding import*

    val_dataset = MOSIDataset()
    train_dataset = MOSIDataset('train')
    test_dataset = MOSIDataset('test')

    val_instances = [val_dataset.getdata(i) for i in tqdm(range(val_dataset.length()))]
    train_instances = [train_dataset.getdata(i) for i in tqdm(range(train_dataset.length()))]
    test_instances = [test_dataset.getdata(i) for i in tqdm(range(test_dataset.length()))]

    val_embeds = getembeds(val_instances, model)
    train_embeds = getembeds(train_instances, model)
    test_embeds = getembeds(test_instances, model)

    params, (accuracies,sparse_cnts) = get_sparse_linear_model(model, train_embeds, val_embeds, 
                                                               test_embeds, modelsavedir='ckpt/mosisparselinearmodel.pt')
                                         
    '''