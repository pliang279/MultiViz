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
feature_id = 5

if not os.path.isdir('private_test_scripts/mosei_simexp/mosei'+str(idx)):
    os.mkdir('private_test_scripts/mosei_simexp/mosei'+str(idx))

# Gradient heatmap for text, vision, and audio
visualize_grad_sparse(dataset, model, data_idx, feature_id)

# Video visualization for vision
process_data(dataset, model, data_idx, feat=feature_id)