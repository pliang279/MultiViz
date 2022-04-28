import torch
from models.analysismodel import analysismodel
from torch.utils.data import DataLoader
import sys
import numpy as np
import torchvision.transforms as transforms


class IMDb_LF(analysismodel):
    def __init__(
        self,
        pretrained_model_path,
        multibench_path,
        bert,
        bert_tokenizer,
        vgg16,
        device="cuda:0",
        batch_size=32,
    ):
        sys.path.insert(2, multibench_path)
        self.model = torch.load(pretrained_model_path).to(device)
        self.model.eval()
        self.modalitynames = ["text", "image"]
        self.modalitytypes = ["text", "image"]
        self.device = device
        self.batch_size = batch_size
        self.bert = bert
        self.bert.eval()
        self.bert_tokenizer = bert_tokenizer
        self.vgg16 = vgg16
        self.vgg16.eval()
        self.image_preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def getunimodaldata(self, datainstance, modality):
        return datainstance[self.modalitynames.index(modality)]

    def getcorrectlabel(self, datainstance):
        # currently returning the last correct genre
        # TODO: Discuss how to fix this
        return datainstance[-1].nonzero()[0][-1]

    def forward(self, datainstance):
        return self.forwardbatch([datainstance])[0]

    def forwardbatch(self, datainstances):
        loader = DataLoader(
            datainstances, num_workers=0, batch_size=self.batch_size, shuffle=False
        )
        outs = []
        for j in loader:

            batch_sz = j[1].shape[0]

            # text
            text_features = torch.zeros((batch_sz, 768)).to(self.device)
            for idx in range(batch_sz):
                raw_text = j[0][idx]
                encoded_input = self.bert_tokenizer(
                    raw_text, return_tensors="pt", truncation=True
                ).to(self.device)
                with torch.no_grad():
                    text_feature = self.bert(**encoded_input)[0].squeeze().mean(0)
                text_features[idx] = text_feature

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

    def getlogit(self, resultobj):
        return resultobj[0]

    def getprelinear(self, resultobj):
        return resultobj[1]

    def getpredlabel(self, resultobj):
        return resultobj[0].argmax(-1).item()

    def getprelinearsize(self):
        return 1024

    def replaceunimodaldata(self, datainstance, modality, newdata):
        if modality == "image":
            ret = (datainstance[0], newdata, datainstance[2])
            assert len(ret) == len(datainstance)
            return ret
        elif modality == "text":
            ret = (newdata, datainstance[1], datainstance[2])
            assert len(ret) == len(datainstance)
            return ret
        else:
            raise ValueError(f"{modality} not compatible with this model")

    def getgrad(self, datainstance, target):

        # text
        text_grads = None
        token_embeds = None

        def bert_bwd_fn(module, grad_in, grad_out):
            nonlocal text_grads
            text_grads = grad_out[0].squeeze().detach().cpu()

        def bert_fwd_hook(module, input, output):
            nonlocal token_embeds
            token_embeds = output[0].squeeze().detach().cpu()

        bert_bwd_handle = self.bert.embeddings.word_embeddings.register_backward_hook(
            bert_bwd_fn
        )
        bert_fwd_handle = self.bert.embeddings.word_embeddings.register_forward_hook(
            bert_fwd_hook
        )
        text_features = np.zeros((1, 768))
        raw_text = datainstance[0]
        encoded_input = self.bert_tokenizer(
            raw_text, return_tensors="pt", truncation=True
        ).to(self.device)
        text_features = self.bert(**encoded_input)[0].squeeze().mean(0).unsqueeze(0)
        bert_fwd_handle.remove()

        # image
        input_tensors = torch.zeros((1, 3, 224, 224))
        input_tensors[0] = self.image_preprocess(datainstance[1])
        image_features = None

        def hook(module, input, output):
            nonlocal image_features
            image_features = input[0]

        handle = self.vgg16.classifier[6].register_forward_hook(hook)
        input_tensors.requires_grad = True
        _ = self.vgg16(input_tensors.to(self.device))
        handle.remove()

        # fprop through LF and backprop
        lf_model_inp = [text_features.float(), image_features.float()]
        out = self.model(lf_model_inp)
        out[0][target].backward()
        bert_bwd_handle.remove()

        ret_raw_inp = [token_embeds.squeeze(), input_tensors.squeeze()]
        grads = [text_grads, input_tensors.grad.detach().cpu()[0]]

        return ret_raw_inp, grads


def main():
    pretrained_model_path = (
        "/home/paul/multimodal_analysis/structured_framework/visuals/best_lf.pth"
    )
    multibench_path = "/home/paul/MultiBench"
    imdb_lf = IMDb_LF(
        pretrained_model_path, multibench_path, device="cuda", batch_size=128
    )
    return


if __name__ == "__main__":
    main()
