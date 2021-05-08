
import csv
import os
import logging
import argparse
import random


import numpy as np
import torch
import torch.nn.functional as F

from transformers.tokenization_bert import BertTokenizer  # train_cbert_and_augment()에서 호출
from transformers.modeling_bert import BertForMaskedLM    # train_cbert_and_augment()에서 호출

from data_processors import get_task_processor  # train_cbert_and_augment()에서 호출



#---------------------------------------------------------------------------------------------#
''' 초기 설정 '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용

BERT_MODEL = 'bert-base-uncased'  # pre-trained model 호출을 위한 '모델명' 문자열 저장 -> BertTokenizer.from_pretrained('bert-base-uncased')

# 로그 띄우기 (logg level = 로그의 심각한 정도)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
#---------------------------------------------------------------------------------------------#
''' class InputFeatures '''



#---------------------------------------------------------------------------------------------#
''' convert_examples_to_features '''

# 입력데이터 받고 두 가지 task 수행하는 부분?

# 강필성교수님 BERT 강의 13분 30초부터 쭉 보고 이 파티 코드리뷰하기
# input data를 3가지로 convert하는 부분(?) - 
# task 1 : 15% (=0.15)만큼 mask하는 부분 (masked_lm_prob)
# task 2 : Next Sentence Prediction (NSP)하는 부분

#---------------------------------------------------------------------------------------------#
''' prepare_data - Input Data 구성하는 부분(?) '''
                                                                                        # BERT의 INPUT Format
def prepare_data(features):                                                             # [CLS], [SEP]
    all_init_ids = torch.tensor([f.init_ids for f in features], dtype=torch.long)       # 모든 index
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)     # BERT Tokenizer의 토큰 ID
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)   # mask ID (뭐가 봐도 되는 토큰이고 아닌지)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long) # segment ID (문장 구분)
    all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_init_ids, all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels)
    # TensorDataset()을 통해 각 텐서들은 init_index를 따라 인덱싱된다
    return tensor_data

#---------------------------------------------------------------------------------------------#
''' rev_wordpiece '''



#---------------------------------------------------------------------------------------------#
''' main '''

def main():
    parser = argparse.ArgumentParser()

    ## Requierd parameters
    parser.add_argument("--data_dir", default="datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="aug_data", type=str,
                        help="The output dir for augmented dataset")

    parser.add_argument("--task_name",default="subj",type=str,
                        help="The name of the task to train.")

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument('--cache', default="transformers_cache", type=str)  # 캐시


    ''' < prepend setting 적용 > 배치사이즈 8 // lr 4e-5 // epochs 10.0  
        < 참고로 expand setting은 > epochs를 150.0으로 해야 converge함   '''
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    # learning rate를 linear하게 증가시킴
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--sample_num', type=int, default=1,
                        help="sample number")
    parser.add_argument('--sample_ratio', type=int, default=7,
                        help="sample ratio")
    parser.add_argument('--gpu', type=int, default=0,
                        help="gpu id")
    parser.add_argument('--temp', type=float, default=1.0,
                        help="temperature")

    args = parser.parse_args()

    print(args)
    train_cbert_and_augment(args)  # train start!
#---------------------------------------------------------------------------------------------#
''' compute_dev_loss '''








#---------------------------------------------------------------------------------------------#
''' augment_train_data '''







#---------------------------------------------------------------------------------------------#
''' train_cbert_and_augment '''
# main() 마지막에서 호출
def train_cbert_and_augment(args):

    ''' task 정의 '''
    task_name = args.task_name                      # 어떤 task할것인지 파라미터로 가져옴 task_name = subj
    os.makedirs(args.output_dir, exist_ok=True)

    ''' 난수 seed 고정 '''
    # -> 동일한 세트의 난수 생성 (https://antilibrary.org/2481) / (https://hoya012.github.io/blog/reproducible_pytorch/)
    random.seed(args.seed)                          # randomness 제어
    np.random.seed(args.seed)                       # np seed
    torch.manual_seed(args.seed)                    # torch seed
    torch.cuda.manual_seed_all(args.seed)           # cuda seed
    torch.backends.cudnn.deterministic = True

    os.makedirs(args.output_dir, exist_ok=True)

    ''' load data & split train and dev data '''
    processor = get_task_processor(task_name, args.data_dir)    # data_dir에서 data가져옴

    label_list = processor.get_labels(task_name)                # label 추출
    train_examples = processor.get_train_examples()             # train set
    dev_examples = processor.get_dev_examples()                 # dev (= validation set =/= test set)


    ''' pretrained BERT 모델(가중치)을 가져온다 ('BERT_MODEL = bert-base-uncased') '''
    ''' [[[[[ 기존에 버트 모델 그냥 가져와서 쓰기만 하면 되는건가??? 따로 공부할 필요 X?? ]]]]] '''
    # 추후에 encoding / embedding할 때 문자열 데이터셋을 모델에 넣어주면 int형 index를 반환함(아마도)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL,
                                              do_lower_case=True,
                                              cache_dir=args.cache)
    # Masked Language Model(MLM) 
    model = BertForMaskedLM.from_pretrained(BERT_MODEL,
                                            cache_dir=args.cache)

    ''' input data 중 token embeddings 설정(?) (https://wikidocs.net/64779) '''
    if len(label_list) > 2:
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(len(label_list), 768)  # (임베딩할 단어 수, 임베딩할 벡터의 차원)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)  # 가중치를 평균 0, 편차 0.02로 초기화

    model.to(device)  # 모델 호출


    ''' [ train data ] '''
    # load한 데이터 중 train set과 label list를 convert()한다!!
    train_features = convert_examples_to_features(train_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    # prepare_data() 호출하여 convert한 데이터를 ~~~
    train_data = prepare_data(train_features)
    # torch.utils.data의 RandomSampler와 DataLoader를 사용하여 ~~~    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size)


    ''' [ dev data(=validation data) ] '''
    # load한 데이터 중 validation set과 label list를 convert()한다!!
    dev_features = convert_examples_to_features(dev_examples, label_list,
                                                  args.max_seq_length,
                                                  tokenizer, args.seed)
    # prepare_data() 호출하여 convert한 데이터를 ~~~
    dev_data = prepare_data(dev_features)
    # torch.utils.data의 RandomSampler와 DataLoader를 사용하여 ~~~    
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.train_batch_size)








#---------------------------------------------------------------------------------------------#
'''  '''


if __name__ == "__main__":
    main()

