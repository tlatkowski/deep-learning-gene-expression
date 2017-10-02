# deep-learning-gene-expression
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

# Model Flow
The below diagram depicts the training and testing procedures:

![](pics/model_flow.png)

# Dataset

The dataset is publicity available and was downloaded from [GEO (NCBI) repository](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS4431). Data file in this repository was cleaned up and contains only raw data with annotated genes and gene sequences annotations.
