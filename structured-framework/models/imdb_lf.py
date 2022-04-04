import torch
from models.analysismodel import analysismodel
import copy
import sys

class IMDb_LF(analysismodel):

    def __init__(self, pretrained_model_path, multibench_path, device='cuda:0', batch_size=32):
        sys.path.insert(2, multibench_path)
        self.model = torch.load(pretrained_model_path).to(device)
        self.modalitynames = ['image','audio','text']
        self.modalitytypes = ['timeseries','timeseries','timeseries']
        self.device=device
        self.batch_size = batch_size

    def getunimodaldata(self,datainstance,modality):
        raise NotImplementedError
        return datainstance[self.modalitynames.index(modality)]

    def getcorrectlabel(self,datainstance):
        raise NotImplementedError
        return datainstance[-1]

    def forward(self, datainstance):
        raise NotImplementedError
        return self.forwardbatch([datainstance])[0]

    def forwardbatch(self,datainstances):
        raise NotImplementedError

    def getlogitsize(self):
        raise NotImplementedError

    def getlogit(self,resultobj):
        raise NotImplementedError

    def getprelinear(self,resultobj):
        raise NotImplementedError

    def getpredlabel(self,resultobj):
        raise NotImplementedError

    def getprelinearsize(self):
        raise NotImplementedError

    def replaceunimodaldata(self,datainstance,modality,newdata):
        raise NotImplementedError



def main():
    pretrained_model_path = '/home/paul/multimodal_analysis/structured_framework/visuals/best_lf.pth'
    multibench_path = '/home/paul/MultiBench'
    imdb_lf = IMDb_LF(pretrained_model_path, multibench_path, device='cuda', batch_size=128)
    return


if __name__ == '__main__':
    main()

