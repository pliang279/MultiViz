import torch
from models.analysismodel import analysismodel
from torch.utils.data import DataLoader
import sys
import numpy as np
import torchvision.transforms as transforms

class IMDb_LF(analysismodel):

    def __init__(self, pretrained_model_path, multibench_path, word2vec, vgg16, device='cuda:0', batch_size=32):
        sys.path.insert(2, multibench_path)
        self.model = torch.load(pretrained_model_path).to(device)
        self.model.eval()
        self.modalitynames = ['text', 'image']
        self.modalitytypes = ['text', 'image']
        self.device = device
        self.batch_size = batch_size
        self.word2vec = word2vec
        self.vgg16 = vgg16
        self.vgg16.eval()
        self.image_preprocess = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225]),
                                ])

    def getunimodaldata(self, datainstance, modality):
        return datainstance[self.modalitynames.index(modality)]

    def getcorrectlabel(self,datainstance):
        # currently returning the last correct genre
        # TODO: Discuss how to fix this
        return datainstance[-1].nonzero()[0][-1]

    def forward(self, datainstance):
        return self.forwardbatch([datainstance])[0]

    def forwardbatch(self, datainstances):
        loader = DataLoader(datainstances, num_workers=0, batch_size=self.batch_size, shuffle=False)
        outs = []
        for j in loader:
            
            batch_sz = j[1].shape[0]

            # text
            text_features = np.zeros((batch_sz, 300))
            for idx in range(batch_sz):
                raw_text = j[0][idx]
                words = raw_text.split()
                if len([self.word2vec[w] for w in words if w in self.word2vec]) == 0:
                    text_features[idx, :] = np.zeros((300,))
                else:
                    text_features[idx, :] = np.array([self.word2vec[w] for w in words if w in self.word2vec]).mean(axis=0)
            text_features = torch.from_numpy(text_features).to(self.device)

            # images
            input_tensors = torch.zeros((batch_sz, 3, 224, 224))
            for idx in range(batch_sz):
                input_tensors[idx] = self.image_preprocess(j[1][idx].numpy())
            image_features = None
            def hook(module, input, output):
                nonlocal image_features
                image_features = input[0]
            handle = self.vgg16.classifier[6].register_forward_hook(hook)
            with torch.no_grad():
                _ = self.vgg16(input_tensors.to(self.device))
            handle.remove()

            model_inp = [text_features.float(), image_features.float()]

            # fprop through LF model
            model_features = None
            def hook(module, input, output):
                nonlocal model_features
                model_features = input[0].cpu()
            handle = self.model.head.fc.register_forward_hook(hook)
            with torch.no_grad():
                out = self.model(model_inp)
            
            for i in range(len(j[0])):
                outs.append((out[i], model_features[i]))
            handle.remove()
                
        return outs

    def getlogitsize(self):
        return 23

    def getlogit(self,resultobj):
        return resultobj[0]

    def getprelinear(self,resultobj):
        return resultobj[1]

    def getpredlabel(self,resultobj):
        return resultobj[0].argmax(-1).item()

    def getprelinearsize(self):
        return 1024

    def replaceunimodaldata(self, datainstance, modality, newdata):
        if modality == 'image':
            ret = (datainstance[0], newdata, datainstance[1])
            assert len(ret) == len(datainstance)
            return ret
        else:
            raise NotImplementedError


def main():
    pretrained_model_path = '/home/paul/multimodal_analysis/structured_framework/visuals/best_lf.pth'
    multibench_path = '/home/paul/MultiBench'
    imdb_lf = IMDb_LF(pretrained_model_path, multibench_path, device='cuda', batch_size=128)
    return


if __name__ == '__main__':
    main()

