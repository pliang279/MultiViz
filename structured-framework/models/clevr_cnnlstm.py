import os
import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(1, "/home/anon/anon/multimodal_analysis/structured-framework/models/clevr-iep")

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize

import iep.utils as utils
import iep.programs
from iep.data import ClevrDataset, ClevrDataLoader
from iep.preprocess import tokenize, encode
from cnn_lstm import build_cnn, load_vocab

from models.analysismodel import analysismodel
from datasets.clevr import CLEVRDataset



class CLEVRCNNLSTM(analysismodel):
    def __init__(self, device="cuda"):
        super(analysismodel, self).__init__()
        self.device = device
        self.path = 'models/clevr-iep/models/CLEVR/cnn_lstm.pt'
        self.model, _ = utils.load_baseline(self.path)
        self.vocab = load_vocab(self.path)
        self.dtype = torch.cuda.FloatTensor
        self.cnn = build_cnn(self.dtype)
        self.modalitynames = ["image", "text"]
        self.modalitytypes = ["image", "text"]
        self.answermapping = self.vocab['answer_idx_to_token']

    def pred_answer(self, outputs):
        idx = 0
        ans_type = int(outputs["pred_answer_type"].argmax(-1))
        if ans_type == 0:
            idx = int(not (outputs["pred_answer_binary"].sigmoid() > 0.5))
        elif ans_type == 1:
            idx = int(outputs["pred_answer_attr"].argmax(-1)) + 2
        else:
            idx = int(outputs["pred_answer_reg"].argmax(-1)) + 17
        return idx

    def getunimodaldata(self, datainstance, modality):
        if modality == "image":
            return np.asarray(PIL.Image.open(datainstance[0]).convert("RGB"))
        elif modality == "text":
            return datainstance[1]
        else:
            raise ValueError

    def getcorrectlabel(self, datainstance):
        return datainstance[3]

    # Helper Functions #
    def process_image(self, img):
        img_size = (224, 224)
        img = imresize(img, img_size, interp='bicubic')
        img = img.transpose(2, 0, 1)[None]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        img = (img.astype(np.float32) / 255.0 - mean) / std
        return img

    def get_question_var(self, question):
        question_tokens = tokenize(question,
                            punct_to_keep=[';', ','],
                            punct_to_remove=['?', '.'])
        question_encoded = encode(question_tokens,
                            self.vocab['question_token_to_idx'],
                            allow_unk=True)
        question_encoded = torch.LongTensor(question_encoded).view(1, -1)
        question_encoded = question_encoded.type(self.dtype).long()
        question_var = question_encoded
        return question_var
    # End Helper Functions #   

    def forward(self, datainstance):
        image = datainstance[0]
        question = datainstance[1]

        # Load and preprocess the image
        img = imread(image, mode='RGB')
        img = self.process_image(img)

        # Use CNN to extract features for the image
        img_var = torch.FloatTensor(img).type(self.dtype)
        feats_var = self.cnn(img_var)

        # Tokenize the question
        question_var = self.get_question_var(question)

        # Run the model
        self.model.type(self.dtype)

        # get prelinear features
        model_features = None

        def hook(module, input, output):
            nonlocal model_features
            model_feat = input
            model_features = model_feat[0][0]
        handle = self.model.classifier[4].register_forward_hook(hook)    

        logits = self.model(question_var, feats_var)

        handle.remove()

        return logits[0].detach(), model_features.detach()

    # in this case we don't do batching, so we just do one at a time:
    def forwardbatch(self, datainstances):
        outs = []
        for di in datainstances:
            outs.append(self.forward(di))
        return outs

    def getlogitsize(self):
        return 32

    def getlogit(self, resultobj):
        return resultobj[0]

    def getprelinear(self, resultobj):
        return resultobj[1]

    def getpredlabel(self, resultobj):
        return resultobj[0].argmax().item()

    def getpredanswer(self, predlabel):
        return self.answermapping[predlabel]

    def getprelinearsize(self):
        raise 1024

    def replaceunimodaldata(self, datainstance, modality, newinput):
        if modality == "image":
            randname = "tmp/clevr" + str(random.randint(0, 100000000)) + ".jpg"
            plt.imsave(randname, newinput)
            return randname, datainstance[1], datainstance[2], datainstance[3]
        elif modality == "text":
            return datainstance[0], newinput, datainstance[2], datainstance[3]
        else:
            raise ValueError

    def getgrad(self, datainstance, target, prelinear=False):
        self.model.zero_grad()
        imgfile = datainstance[0]
        img = imread(imgfile, mode='RGB')
        img = self.process_image(img)

        img_var = torch.FloatTensor(img).type(self.dtype)
        img_var.requires_grad = True
        feats_var = self.cnn(img_var)

        question = datainstance[1]
        question_var = self.get_question_var(question)
        print(question_var)


        if prelinear:
            model_features = None

            def hook(module, input, output):
                nonlocal model_features
                model_feat = input
                model_features = model_feat[0][0]
            handle = self.model.classifier[4].register_forward_hook(hook)    

        logits = self.model(question_var, feats_var)

        if prelinear:
            model_features[target].backward()
        else:
            logits[0][target].backward()
        grad = img_var.grad.detach()

        if prelinear:
            handle.remove()
        return img_var, grad, imgfile          

    def getgradtext(self, datainstance, target, prelinear=False, alltarget=False):
        self.model.zero_grad()
        imgfile = datainstance[0]
        img = imread(imgfile, mode='RGB')
        img = self.process_image(img)

        img_var = torch.FloatTensor(img).type(self.dtype)
        img_var.requires_grad = True
        feats_var = self.cnn(img_var)

        question = datainstance[1]
        question_var = self.get_question_var(question)

        text_embedding = None
        text_ids = None
        gradd = None

        def hook_forward(module, input, output):
            nonlocal text_embedding, text_ids
            text_embedding = output[0]
            text_ids = input[0]

        def hook_backward(module, input, output):
            nonlocal gradd
            gradd = output[0][0]

        handle = self.model.rnn.embed.register_forward_hook(
            hook_forward
        )
        handle22 = self.model.rnn.embed.register_backward_hook(
            hook_backward
        )

        if prelinear:
            model_features = None

            def hook(module, input, output):
                nonlocal model_features
                model_feat = input
                model_features = model_feat[0][0]
            handle1 = self.model.classifier[4].register_forward_hook(hook)    

        logits = self.model(question_var, feats_var)

        if alltarget:
            torch.sum(logits[0]).backward(create_graph=True)
        elif prelinear:
            model_features[target].backward()
        else:
            logits[0][target].backward()

        res = torch.sum(text_embedding * gradd, dim=1)

        handle.remove()
        handle22.remove()
        if prelinear:
            handle1.remove()

        return res, self.parse(question), img_var, text_ids

    def getdoublegrad(self, datainstance, target, targetwords, alltarget=True):
        with torch.backends.cudnn.flags(enabled=False):
            res, di, img_var, text_ids = self.getgradtext(
                datainstance, target, alltarget=alltarget
            )
            ac = 0.0
            for id in targetwords:
                ac += torch.abs(res[id])  
            rets = torch.autograd.grad(ac, img_var)
        return rets[0], di, text_ids    
        
    def parse(self, sent):
        words = []
        for word in sent[:-1].split(" "):
            words.append(word)
        words.append("?")
        words.append("<end>")
        words.insert(0, "<start>")
        return words

if __name__ == '__main__':
    dataset = CLEVRDataset()
    model = CLEVRCNNLSTM()

    datainstance = dataset.getdata(0)
    resobj = model.forward(datainstance)    
    pred_label = model.getpredlabel(resobj)
    pred_answer = model.getpredanswer(pred_label)
    prelinear = model.getprelinear(resobj)
    
    print(pred_label, pred_answer)        
    print(prelinear.shape)
    print(model.getgrad(datainstance, pred_label))
    print(model.getgradtext(datainstance, pred_label))
    print(model.getdoublegrad(datainstance, pred_label, [0]))
    
