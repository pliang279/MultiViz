# MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models

This repository contains code and experiments for performing interpretability analysis in a multimodal setting.

[MultiViz website](https://andy-xingbowang.com/multivizSim/)

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - [Yiwei Lyu](https://github.com/lvyiwei1) (ylyu1@andrew.cmu.edu)
  - [Gunjan Chhablani](https://gchhablani.github.io/) (chhablani.gunjan@gmail.com)
  - [Nihal Jain](https://nihaljn.github.io/) (nihalj@cs.cmu.edu)
  - Zihao Deng (zihaoden@andrew.cmu.edu)
  - [Xingbo Wang](https://andy-xingbowang.com/) (xingbo.wang@connect.ust.hk)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/) (morency@cs.cmu.edu)
  - [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) (rsalakhu@cs.cmu.edu)

## Code framework for visualization of multimodal model analysis

We designed a structured framework that allows easy analysis and visualization of multimodal models. To run anything within the structured framework, you should have ``structured-framework/`` as your working directory.

In the structured framework, there are 4 main modules: datasets, models, analysis and visualizations. 

Each script in the datasets module loads data points from a specific dataset, and also interfaces all necessary information that directly comes from the dataset (such as label-id-to-answer mappings). To add a new dataset to the framework, simply add another file under this module following the same format as the existing ones.

Each script in the models module contains a wrapper class for a specific multimodal model for a particular dataset. All such classes should be subclass of the AnalysisModel class defined in ``models/analysismodel.py``, which specifies certain functionalities the class must implement such as ``forward``, ``getpredlabel``, etc. To add a new model to the framework, add another file under this module and write a wrapper class for your model that extends the AnalysisModel class.

Under the analysis module there are scripts that runs various analysis methods on arbitrary dataset and model combinations. This is enabled by calling on common functionalities specified in AnalysisModel class. These scripts outputs raw analysis results (usually just a bunch of numbers), and scripts in the visualizations module are tools to create visualizations of these raw results. To add additional analysis and visualization methods, simply add some functions to these modules.




