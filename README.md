# Text Augmentation using by BERT model
Keyword : #NLP, #BERT, #TextAugmentation, #DataAugmentation #EDA
##### Written by Taehyeon Kwon  (Reporting date : 2021.04 ~ 2021.06)

---
---
## Abstract

* This code is originally released from amazon-research package (https://github.com/amazon-research/transformers-data-augmentation) In the paper, we mentioned https://github.com/varinf/TransformersDataAugmentation url so we are providing a copy of the same code here.

* Code associated with the Data Augmentation using Pre-trained Transformer Models paper

* Code contains implementation of the following data augmentation methods

  - EDA
  - Backtranslation
  - BERT


---
---
## Dependencies

* To run this code, you need following dependencies "in your new virtual environment".
  - python 3.7         ( $ conda create --name tae python=3.7 )
  - fairseq 0.9        ( $ conda install -c conda-forge fairseq=0.9.0 )
  - Pytorch 1.5        ( $ conda install -c conda-forge pytorch=1.5.0 )
     - Download as appropriate for your GPU.
  - transformers 2.9   ( $ conda install -c conda-forge transformers=2.9.0 )

* Then, check your dependencies.
  - $ conda list
  

---
---
## DataSets

* I used a dataset from following resource to test augmentationing code using by bert.
  - TREC : https://github.com/1024er/cbert_aug/tree/crayon/datasets/TREC
  - Augmentationing code : https://github.com/varinf/TransformersDataAugmentation

* Low-data setup
  - L2+L3 DataSets : For students who speak Korean as first language, this dataset consists of sentences for students studying English as a second language.
  - [Private]

* If you want to augment your datasets (anything text data), check "How to run - Prepare your datasets"
  
  
---
---
## Modify the path in script file for suit yours

* src/scripts/bert_L2_lower.sh or src/scripts/bert_trec_lower.sh
  - SRC
  - CACHE
  - RAWDATADIR


---
---
## How to run

* For prepare a source code, clone or download file in this repository.
  - !git clone https://github.com/taehyeon96/TextAugmentation_BERT.git

* Prepare the L2-Datasets(or your datasets) and match data format
  - check above [Datasets]
  - .tsv file
  - labeling : "Description"
  - [train || dev || test] -> (train+dev) : test = 7.5 : 2.5
  - Put the train.tsv, dev.tsv, test.tsv in the folder (Datasets/L2/)
  - Split each of the .tsv files and put them in each folder (Datasets/L2/exp_{i}_10,  i=[0:14])
    
* To run text augmentation experiment of your dataset, run following jupyter file in this repository.
  - run "start_L2 Aug_with_jupyter.ipynb" 
  
* To download text augmentation result of your datasets, run following jupyter file in this repository.
  - run "download_L2 Aug_with_jupyter.ipynb" 


