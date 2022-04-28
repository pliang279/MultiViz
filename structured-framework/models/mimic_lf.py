import torch
from torch.utils.data import DataLoader
from models.analysismodel import analysismodel
import copy
import sys


class MIMICLF(analysismodel):
    def __init__(
        self, pretrained_model_path, multibench_path, device="cuda:0", batch_size=32
    ):
        sys.path.insert(2, multibench_path)
        self.model = torch.load(pretrained_model_path).to(device)
        self.modalitynames = ["static", "timeseries"]
        self.modalitytypes = ["tabular", "timeseriesC"]
        self.device = device
        self.batch_size = batch_size

    def getunimodaldata(self, datainstance, modality):
        return datainstance[self.modalitynames.index(modality)]

    def getcorrectlabel(self, datainstance):
        return datainstance[-1]

    def forward(self, datainstance):
        return self.forwardbatch([datainstance])[0]

    def forwardbatch(self, datainstances):
        loader = DataLoader(
            datainstances, num_workers=0, batch_size=self.batch_size, shuffle=False
        )
        outs = []
        with torch.no_grad():
            for j in loader:
                model_features = None

                def hook(module, input, output):
                    nonlocal model_features
                    model_features = input[0]

                handle = self.model.head.fc2.register_forward_hook(hook)
                out = self.model([jj.float().to(self.device) for jj in j[:-1]])
                for i in range(len(j[0])):
                    outs.append((out[i], model_features[i]))
                handle.remove()
        return outs

    def getlogitsize(self):
        return 2

    def getlogit(self, resultobj):
        return resultobj[0]

    def getprelinear(self, resultobj):
        return resultobj[1]

    def getpredlabel(self, resultobj):
        return resultobj[0].argmax(-1).item()

    def getprelinearsize(self):
        return 40

    def replaceunimodaldata(self, datainstance, modality, newdata):
        c = copy.deepcopy(datainstance)
        c[self.modalitynames.index(modality)] = newdata
        return c

    def getgrad(self, datainstance, target):
        inputer = [
            torch.FloatTensor(j).unsqueeze(0).to(self.device) for j in datainstance[:-1]
        ]
        for sample in inputer:
            sample.requires_grad = True
        """
        modalids = [self.modalitynames.index(modality) for modality in modalities]
        samples = [torch.autograd.Variable(inputer[modalid],requires_grad=True) for modalid in modalids]
        for i in range(len(samples)):
            inputer[modalids[i]] = samples[i]
        """
        out = self.model(inputer)
        out[0][target].backward()
        grads = [sample.grad.detach()[0] for sample in inputer]
        return inputer, grads
