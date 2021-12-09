# Multimodal Analysis
This repository contains code and experiments for performing interpretability analysis in a multimodal setting.

## Methods & Usage
### EMAP

```python
import numpy as np
from mma.analysis.metrics.emap import Emap

# This can be a list of numpy arrays, a dict of numpy arrays or a dict of dict of numpy arrays.
# It is assumed that the predictor function takes in the keys of the dictionary.

dataset = {
  'visual_inputs': {
    'features': all_image_features,
    'normalized_boxes': all_normalized_boxes
  },
  'textual_inputs': {
    'input_ids': all_text_input_ids,
    'attention_mask': all_text_attention_masks,
    'token_type_ids': all_text_token_type_ids
  }
}

def predictor_fn(visual_inputs, textual_inputs):
  ...

emap = Emap(predictor_fn, dataset)

emap_scores = emap.compute_emap_scores(batch_size=4) # Computers Emap Logit Scores
orig_scores = emap.compute_predictions('orig', batch_size=4) # Compute Original Logit Scores

# Text
orig_score = accuracy_score(orig_labels, orig_preds)
emap_score = accuracy_score(orig_labels, emap_preds)
```
