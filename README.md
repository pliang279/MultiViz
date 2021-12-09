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

### LIME
#### Unimodal

```python
import numpy as np
from mma.analysis.surrogates.lime.lime import Lime

image = ...
text = ...
label_idx = ...

def text_predictor_fn(texts):
  ...
def image_predictor_fn(images):
  ...

# Image
image_init_params = {} # See lime parameters
image_explanation_params = {
  'top_labels': 1,
  'hide_color': 0,
  'num_samples': 30,
  'batch_size': 5
} # See lime parameters

image_exp = Lime.explain_image_instance(
  image_predictor_fn,
  np.array(image),
  image_init_params,
  image_explanation_params
)

temp, mask = image_exp.get_image_and_mask(
  image_exp.top_labels[0],
  positive_only=False,
  num_features=10,
  hide_rest=False
)

img_boundary = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundary)
plt.show()

# Text

text_init_params = {
  'class_names': class_names # list of class names in order
}

text_explanation_params = {
  'num_features': 5,
  'num_samples': 100,
  'top_labels': 1
}

text_exp = Lime.explain_text_instance(
   text_predictor_fn,
   text,
   text_init_params,
   text_explanation_params
)

fig = text_exp.as_pyplot_figure(label=label_idx)
plt.show()
```
