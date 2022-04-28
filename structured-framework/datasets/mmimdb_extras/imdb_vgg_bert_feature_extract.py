import os
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import json
import h5py
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel


data_path = "/home/paul/yiwei/MultiBench/multimodal_imdb.hdf5"
dataset = h5py.File(data_path, "r")
raw_imdb_root_path = "/home/paul/nihalj/MultiBench/datasets/imdb/mmimdb/dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"
vgg16_model = torchvision.models.vgg16(pretrained=True).to(device)
vgg16_model.eval()

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", model_max_length=512, truncation_size="right"
)
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()


def _process_data(filename, root_path):

    data = {}
    filepath = os.path.join(root_path, filename)
    data["imdb_id"] = filename

    # process image
    with Image.open(filepath + ".jpeg") as f:
        raw_img = f.convert("RGB")
        image = np.array(raw_img)
        data["image"] = image
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    input_tensor = preprocess(f.convert("RGB"))
    data["image_tensor"] = input_tensor

    # process text
    with open(filepath + ".json", "r") as f:
        info = json.load(f)
        plot = info["plot"]
        plot_id = np.array([len(p) for p in plot]).argmax()
        data["plot"] = plot[plot_id]

    return data


def main():

    n_samples = len(dataset["imdb_ids"])
    rows = []
    for i in tqdm(range(n_samples)):
        imdb_id = dataset["imdb_ids"][i].decode("utf-8")
        data = _process_data(imdb_id, raw_imdb_root_path)
        labels = dataset["genres"][i]

        words = data["plot"]
        encoded_input = tokenizer(words, return_tensors="pt", truncation=True).to(
            device
        )
        with torch.no_grad():
            text_features = bert_model(**encoded_input)[0].squeeze().mean(0).cpu()

        image_features = None

        def hook(module, input, output):
            nonlocal image_features
            image_features = input[0].squeeze().cpu()

        handle = vgg16_model.classifier[6].register_forward_hook(hook)
        with torch.no_grad():
            _ = vgg16_model(data["image_tensor"].unsqueeze(0).to(device))
        handle.remove()

        rows.append((imdb_id, text_features, image_features, labels))

    cols = ["imdb_id", "text_features", "image_features", "labels"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_pickle("imdb_vgg_bert.pkl")

    return


if __name__ == "__main__":
    main()
