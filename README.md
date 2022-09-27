# MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models

This repository contains code and experiments for performing interpretability analysis in a multimodal setting.

## Contributors

Anonymous

# Code framework for visualization of multimodal model analysis

We designed a structured framework that allows easy analysis and visualization of multimodal models. To run anything within the structured framework, you should have ``structured-framework/`` as your working directory.

## Structure of the code framework

In the structured framework, there are 4 main modules: datasets, models, analysis and visualizations. 

Each script in the datasets module loads data points from a specific dataset, and also interfaces all necessary information that directly comes from the dataset (such as label-id-to-answer mappings). To add a new dataset to the framework, simply add another file under this module following the same format as the existing ones.

Each script in the models module contains a wrapper class for a specific multimodal model for a particular dataset. All such classes should be subclass of the AnalysisModel class defined in ``models/analysismodel.py``, which specifies certain functionalities the class must implement such as ``forward``, ``getpredlabel``, etc. To add a new model to the framework, add another file under this module and write a wrapper class for your model that extends the AnalysisModel class.

Under the analysis module there are scripts that runs various analysis methods on arbitrary dataset and model combinations. This is enabled by calling on common functionalities specified in AnalysisModel class. These scripts outputs raw analysis results (usually just a bunch of numbers), and scripts in the visualizations module are tools to create visualizations of these raw results. To add additional analysis and visualization methods, simply add some functions to these modules.

## Usage: VQA

To download the dataset, you need to download the following things from these urls and unzip all in a folder named ``structured-framework/data``

```
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
```

Here are a few example scripts on running analysis methods on VQA with LXMERT model:

(1) Unimodal LIME: ``structured-framework/examples/vqa_lxmert_unimodallime.py``

(2) EMAP: ``structured-framework/examples/vqa_lxmert_emap.py``

(3) DIME: ``structured-framework/examples/vqa_lxmert_dime.py``

(4) Sparse Linear Model together with local and global representation feature analysis: ``structured-framework/examples/vqa_lxmert_sparselinearmodel.py``

(5) Global representation feature analysis on all features: ``structured-framework/examples/vqa_lxmert_featureonly.py``

(6) No analysis, just showing the original question and image and correct/predicted answer in one image: ``structured-framework/examples/vqa_lxmert_show.py``

Note that the version of LXMERT used is directly from the HuggingFace Transformers pipeline, which contains a bug in image preprocessing that flips the red and blue values of pixels. To use the bug-free version, simply comment out line 571 of ``structured-framework/models/lxmert_extras/utils.py``

## Usage: CLEVR

To download the dataset, run the following command. Then unzip data into a folder named ``structured-framework/data``
```
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
```
We use both MDETR and CNNLSTMSA models on the CLEVR dataset. These are pretrained models from the [MDETR github repository](https://github.com/ashkamath/mdetr) and [clevr-iep github repository](https://github.com/facebookresearch/clevr-iep), respectively. The MDETR model achieves nearly prefect score on the CLEVR dataset, whereas CNNLSTMSA is a baseline proposed in the [original CLEVR paper](https://arxiv.org/abs/1612.06890).

Below are a few example scripts on running analysis methods on CLEVR with MDETR model. To use the CNNLSTMSA model, you should create a new conda environment with python3.5 and install all the requirements from the [clevr-iep github repository](https://github.com/facebookresearch/clevr-iep).

(1) First Order Gradient: ``structured-framework/examples/clevr-mdetr-gradient.py``

(2) Sparse Linear Model together with local and global representation feature analysis:``structured-framework/examples/clevr-mdetr-slm.py``

## Usage: CMU-MOSEI

This repository contains a processed version of CMU-MOSEI validation split in ``structured-framework/data/MOSEI/mosei_valid_processed_features_list.pkl``. 

If you would like to process the data yourself or visualize the analysis results in the form of videos using our provided methods, you need to download the original data from the link posted on the [CMU-MultimodalSDK github](http://immortal.multicomp.cs.cmu.edu/raw_datasets/). Put the downloaded zip file into ``structured-framework/data/``.

This repository also contains a pretrained Multimodal Transformer for CMU-MOSEI in ``structured-framework/models/mult/mosei_sentimult_MULT_a_0.3_e_0.2_o_0.3_res_0.1.pt``. For more information about the MulT model and its training, refer to the [original github repository](https://github.com/yaohungt/Multimodal-Transformer).

Here are a few example scripts on running analysis methods on CMU-MOSEI with MulT model:

(1) First Order Gradient: ``structured-framework/examples/mosei_mult_gradient.py``

(2) Sparse Linear Model together with local and global representation feature analysis: ``structured-framework/examples/mosei_mult_sparselinearmodel.py``

(3) Second Order Gradient: ``structured-framework/examples/mosei_mult_secondordergradient.py``

(4) Generate all analysis results and the json file: ``structured-framework/examples/mosei_mult_runandmakejson.py``

## Usage: MM-IMDb

The raw MM-IMDb dataset can be downloaded from [here](https://archive.org/download/mmimdb/multimodal_imdb.hdf5). The `.hdf5` file downloaded can be used directly in the example scripts below.

The analyses presented in the paper use an LRTF model for predicting the genres of movies from their posters and movie synopsis. To train this model on the MM-IMDb data, follow the instructions [here](https://archive.org/download/mmimdb/multimodal_imdb.hdf5).

Here are a few example scripts for running the analyses presented in the paper with the LRTF model on the MM-IMDb dataset:

(1) First & Second Order Gradients: ``structured-framework/examples/imdb_lrtf_gradsaliency.py``

(2) Sparse Linear Model training and visualization: ``structured-framework/examples/imdb_lrtf_sparselinearmodel.py``

(3) Unimodal LIME: ``structured-framework/examples/imdb_lrtf_unimodallime.py``


## Usage: Flickr30k

To download the Flickr30k dataset, follow the instructions shared on the [official page](http://shannon.cs.illinois.edu/DenotationGraph/) provided in the Downloads section. If you are interested in evaluating the cross-modal interpretations against ground-truth annotations, the Flickr30k Entities annotations can be found on [this GitHub repository](https://github.com/BryanPlummer/flickr30k_entities). 

Then unzip data into a folder named ``structured-framework/data``. It should have the following structure:

```
│   ├── data
│   │   └── flickr30k
│   │       ├── Annotations
│   │       ├── flickr30k-images
│   │       └── Sentences
```

The `Annotations` and `Sentences` folders come from the Flickr30k Entities dataset.

We use both [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) and [ViT](https://huggingface.co/dandelin/vilt-b32-finetuned-flickr30k) models on the Flickr30k dataset, which are present on the HuggingFace model ecosystem. The ViLT model is fine-tuned on the Flicrk30k dataset.

Below are a few example scripts on running analysis methods on Flickr30k with ViLT model.

(1) First Order Gradient: ``structured-framework/examples/flickr30k_vilt_gradsaliency.py``

(2) Second Order Gradient: ``structured-framework/examples/flickr30k_vilt_gradient.py``

There are examples for the CLIP model present as well.

## Usage: MIMIC

To download the dataset, follow the instructions at [MultiBench](https://github.com/pliang279/MultiBench) to obtain the ``im.pk`` file. When loading the dataset in ``structured-framework/datasets/mimic.py``, set the path to the path to ``im.pk``. 

To use the LF model, clone the MultiBench repository from the same link above and then follow instructions in there to train a LF model for MIMIC. Then input both the path to the saved checkpoint and the path to the cloned repository into the constructor of MIMICLF. See the examples below for details.

Here are a few example scripts on running analysis methods on MIMIC with LF model:

(1) Unimodal LIME: ``structured-framework/examples/mimic_lf_unimodallime.py``

(2) EMAP: ``structured-framework/examples/mimic_lf_emap.py``

(3) First Order Gradient: ``structured-framework/examples/mimic_lf_firstordergradient.py``

## Extending the code framework with your own dataset/model/analysis methods.

MultiViz is designed in a way such that it can easily be extended to other datasets, models, and analysis methods.

If you wish to add your own dataset or model and use the existing analysis scripts in the repository, just follow the same format of existing dataset and model classes in this repository. All existing analysis scripts assumes that the model being analyzed implements the functions in ``models/analysismodel.py``, so if you write your model class implementing these functions, you can directly apply existing analysis scripts to your dataset and model.

If you wish to add your own analysis method and make it applicable to multiple existing datasets and models, simply make your analysis function takes a "analysismodel" object as input and only use the functions specified within ``models/analysismodel.py`` to interact with the model, and the resulting script should be applicable to all currently existing datasets and models in this repository.
