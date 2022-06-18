import os
import sys
import torch

sys.path.insert(1, os.getcwd())

from datasets.mosei2 import MOSEIDataset
from models.mosei_mult import MOSEIMULT
from private_test_scripts.mosei_json import*

dataset = MOSEIDataset()
model = MOSEIMULT()

data_idx = 10

if not os.path.isdir('private_test_scripts/mosei_simexp/mosei'+str(data_idx)):
    os.mkdir('private_test_scripts/mosei_simexp/mosei'+str(data_idx))

generate_heatmap_data(data_idx)
generate_video_data(data_idx)
with open('private_test_scripts/mosei_simexp/mosei'+str(idx)+'/mosei'+str(idx)+'.json', 'w') as outfile:
    json.dump(generate_json(idx), outfile, indent=4)