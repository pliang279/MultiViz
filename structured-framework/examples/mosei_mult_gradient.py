import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.mosei2 import MOSEIDataset
from models.mosei_mult import MOSEIMULT
from visualizations.visualizemosei import*
from visualizations.visualizevideo import*

dataset = MOSEIDataset()
model = MOSEIMULT()

data_idx = 10

if not os.path.isdir('private_test_scripts/mosei_simexp/mosei'+str(idx)):
    os.mkdir('private_test_scripts/mosei_simexp/mosei'+str(idx))

# Gradient heatmap for text, vision, and audio
visualize_grad(dataset, model, data_idx)

# Video visualization for vision
process_data(dataset, model, data_idx)
