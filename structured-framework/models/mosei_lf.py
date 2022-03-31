import torch
from torch.utils.data import DataLoader
from models.analysismodel import analysismodel
import copy
import sys
class MOSEILF(analysismodel):
    def __init__(self,pretrained_model_path,multibench_path,device='cuda:0',batch_size=32):
        sys.path.insert(2,multibench_path)
        self.model = torch.load(pretrained_model_path).to(device)
        self.modalitynames = ['image','audio','text']
        self.modalitytypes = ['timeseries','timeseries','timeseries']
        self.device=device
        self.batch_size = batch_size
    def getunimodaldata(self,datainstance,modality):
        return datainstance[self.modalitynames.index(modality)]
    def getcorrectlabel(self,datainstance):
        return datainstance[-1]
    def forward(self,datainstance):
        return self.forwardbatch([datainstance])[0]
    def forwardbatch(self,datainstances):
        loader = DataLoader(datainstances,num_workers=0,batch_size=self.batch_size, shuffle=False)
        outs=[]
        with torch.no_grad():
            for j in loader:
                model_features = None
                def hook(module, input, output):
                    nonlocal model_features
                    model_features = input[0]
                handle = self.model.head.fc2.register_forward_hook(hook)
                out = self.model([jj.float().to(self.device) for jj in j[:-1]])
                handle.remove()
                for i in range(len(j[0])):
                    outs.append((out[i],model_features[i]))
        return outs
    def getlogitsize(self):
        return 2
    def getlogit(self,resultobj):
        return resultobj[0]
    def getprelinear(self,resultobj):
        return resultobj[1]
    def getpredlabel(self,resultobj):
        return resultobj[0].argmax(-1).item()
    def replaceunimodaldata(self,datainstance,modality,newdata):
        c = copy.deepcopy(datainstance)
        c[self.modalitynames.index(modality)] = newdata
        return c



        

