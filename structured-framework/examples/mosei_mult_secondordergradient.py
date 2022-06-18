import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.mosei2 import MOSEIDataset
from models.mosei_mult import MOSEIMULT
from visualizations.visualizemosei import*

dataset = MOSEIDataset()
model = MOSEIMULT()

data_idx = 10
data_instance = dataset.getdata(data_idx)
words = model.getwords(data_instance)[:50]
target_idxs = [2,3,4]
target_words = [words[i] for i in target_idxs]
print(target_words)

# SOG heatmap of vision and audio
visualize_grad_double(dataset, model, idx, target_idxs, reverse=False)