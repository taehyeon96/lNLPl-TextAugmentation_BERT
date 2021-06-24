# Text Augmentation using by BERT model
Keyword : #NLP, #BERT, #TextAugmentation, #DataAugmentation
##### Written by Taehyeon Kwon  (Reporting date : 2021/06)

---
---
## Abstract

* This code is originally released from amazon-research package (https://github.com/amazon-research/transformers-data-augmentation) In the paper, we mentioned https://github.com/varinf/TransformersDataAugmentation url so we are providing a copy of the same code here.

* Code associated with the Data Augmentation using Pre-trained Transformer Models paper

* Code contains implementation of the following data augmentation methods

  - EDA (Baseline)
  - Backtranslation (Baseline)
  - CBERT (Baseline)


---
---
## DataSets

* I used a dataset from following resource to test augmentationing code using by bert.
  - TREC : https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC
  - Augmentationing code : https://github.com/varinf/TransformersDataAugmentation

* Low-data setup
  - L2+L3 DataSets : For students who speak Korean as first language, this dataset consists of sentences for students studying English as a second language.
  - <Private>

  
---
---
## Dependencies

* To run this code, you need following dependencies "in your new virtual environment".
  - python 3.7         ( $ conda create --name tae python=3.7 )
  - Pytorch 1.5        ( $ conda install -c conda-forge pytorch=1.5.0 )
  - fairseq 0.9        ( $ conda install -c conda-forge fairseq=0.9.0 )
  - transformers 2.9   ( $ conda install -c conda-forge transformers=2.9.0 )

* Then, check your dependencies.
  - $ conda list
  

---
---
## How to run

* For prepare a script file, clone the git.   (추후 필요한 것만 따로 빼서 레포에 올릴 것)
  - !git clone https://github.com/varinf/TransformersDataAugmentation.git
  
* For prepare a jupyter file, clone or download file in this repository.
  
* Prepare the L2 Datasets and match data format
  - .tsv file
  - labeling : "Description"
  - [train || dev || test] -> (train+dev) : test = 7.5 : 2.5
  
* To run data augmentation experiment for a given dataset, run "bash" script in scripts folder. For example, to run data augmentation on L2 dataset,
  
  
  
.
