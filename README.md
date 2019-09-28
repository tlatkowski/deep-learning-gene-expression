![](https://img.shields.io/badge/Python-3.6-blue.svg) ![](https://img.shields.io/badge/NumPy-1.14.2-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)

# Deep learning methods for gene expression
Deep learning methods for feature selection in gene expression autism data.
# Description
This project implements several features selection algorithms intended for finding the most significant subset of genes and gene sequences stored in dataset of gene expression microarray. 

Current version of project provides the following list of feature selection algorithms:
* Fisher discriminant analysis
* two sample t-test
* feature correlation with a class
  
More implementation details of the above methods can be found here:

[Data mining for feature selection in gene expression autism data](http://www.sciencedirect.com/science/article/pii/S0957417414005259)

[Feature selection methods in application to gene expression: autism data](http://www.pe.org.pl/articles/2014/8/47.pdf)

The outcome of feature selection stage is consumed by fully connected feedforward neural network. The following list of hyperparameters can be configured in this neural network:
* number of layers,
* number of hidden units in each layer,
* activation function: sigmoid, tanh and ReLU,
* L2 lambda reguralization parameter.
* batch size,
* number of epochs.

# Model Flow
The below diagram depicts the training and testing procedures:

![](pics/model_flow.png)

# Dataset

The dataset is publicity available and was downloaded from [GEO (NCBI) repository](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS4431). Data file in this repository was cleaned up and contains only raw data with annotated genes and gene sequences annotations.

## Dataset details
Number of observations in this dataset equals 146 and number of genes 54613. The database consists of two classes: the first one is related to children with autism (n=82) and the second to control (healthy) children (n=64). Blood draws for all subjects were done between the spring and  summer  of  2004.  Total  RNA  was  extracted  for  microarray experiments with Affymetrix Human U133 Plus 2.0 39 Expression Arrays. 


## Run the pipeline locally

### Installation (Ubuntu)

In order to install all requirements execute the following script:
(If needed add 'execute' permission to *install.sh* script before running it):
```bash
chmod a+x bin/install.sh
```

```bash
./bin/install.sh
```

Then activate the Virtual Environment (if needed):
```bash
source .venv/bin/activate
```
In order to run the pipeline execute:
```
python pipeline.py
```

## Run the pipeline on Google Colab
In order to run the pipeline on Google Colab use the following notebook:
[Deep Learning Gene Expression in Google Colab](https://github.com/tlatkowski/deep-learning-gene-expression/blob/master/colab/deep_learning_feature_selection.ipynb)

## Pipeline configuration

Pipeline gives you possibility to tweak training parameters. In order to modify them use configuration file
placed in `./config/experiment_setup.yml`. Below you can find the default configuration:

```yaml
selection_methods:
  - method: fisher
    num_features: 100
  - method: ttest
    num_features: 100
  - method: corr
    num_features: 100
  - method: random
    num_features: 100
hyperparameters:
  learning_rate: 0.001
  input_size: 100
  hidden_sizes: [80]
  output_size: 1
  num_features: 100
  activation_function: 'tanh'
  lambda_reg: 0.8
  norm_data: True
  data_file: 'data/data.tsv'
training:
  num_epochs: 10000
  cross_validation_folds: 10
  batch_size: 20  # online learning when batch_size=1
```