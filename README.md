# TextAug
Keyword : #NLP, #BERT, #TextAugmentation, #DataAugmentation, #Fine-tuning


### Start : 21/04/05
---
#### ~ 21/04/11 ]
@ Generative Adversarial Networks(GAN) Paper
 - Paper reading
 - GAN에 대한 흐름 파악
 - 추후 연구 과제 : Object function에 대한 깊은 이해 필요
 - https://arxiv.org/abs/1406.2661


@ Text Augmentation Using Hierarchy-based Word Replacement Paper
 - Paper reading
 - 계층구조 기반의 Text Augmentation Method 이해
 - http://koreascience.or.kr/article/JAKO202103440878936.page

---
#### ~ 21/04/18 ]
@ Generative Adversarial Networks(GAN) Paper 보충
 - 일전에 도출된 추후 연구과제에 대한 보충
 - GAN의 Loss function에 대한 흐름 파악 완료
 - https://m.blog.naver.com/euleekwon/221557899873
 - https://m.blog.naver.com/euleekwon/221558014002

@ GAN Paper Review
 - 세부 내용을 이해하기 위한 Paper Review 참고 블로그
 - 완벽한 이해를 하지 못하였으므로, 추후 다시 공부할 예정
 - https://ysbsb.github.io/gan/2020/10/07/Generative-Adversarial-Networks.html

@ Text Augmentation Paper Review
 - 고려대학교_DSBA Lab의 Text Augmentation Paper Reivew
 - Text Augmentation에 관련된 알고리즘 및 다양한 방법론 습득
 - 이를 기반으로 다음주 교수님 Meeting 예정
 - http://dsba.korea.ac.kr/seminar/?mod=document&pageid=1&keyword=text%20aug&uid=1328

---
#### ~ 21/04/25 ]
@ Text Augmentation Paper Review
 - 고려대학교_DSBA Lab의 Text Augmentation Paper Reivew
 - Text Augmentation에 관련된 알고리즘 및 다양한 방법론 습득
 - http://dsba.korea.ac.kr/seminar/?mod=document&pageid=1&keyword=text%20aug&uid=1328

@ 추후 연구과제 도출 1
 - BERT 모델에 대한 정확한 이해를 위해 논문 Reading
 - https://www.researchgate.net/publication/346870334_Data_Augmentation_Using_Pre-trained_Transformer_Models

@ 추후 연구과제 도출 2
 - BERT 모델 구현 (Replication)
   -> CBERT (Baseline)
   -> BERT Prepend (Proposed by this paper)
 - 이를 통해 사용된 메소드, 구조, 하이퍼파라미터 등 파악
 - https://github.com/varinf/TransformersDataAugmentation

---
#### ~ 21/05/02 ]
@ 210425 Meeting에서 도출된 추후 연구과제 1 수행
 - BERT 모델에 대한 정확한 이해를 위해 논문 Reading
 - https://www.researchgate.net/publication/346870334_Data_Augmentation_Using_Pre-trained_Transformer_Models

@ SKT - BERT 모델 심층 이해 (키워드 : Tokenizing, Input Data, Task 종류)
- 논문 이해를 증진하기 위해 모델의 input data와 Tokenizing 방법 그리고
  BERT를 활용한 네 가지 TASK에 대해 조사했다. 
- 이를 기반으로 코드 분석을 수행할 예정.
- https://www.youtube.com/watch?v=riGc8z3YIgQ

@ SKT - BERT 모델 실습 (키워드 : 실습, 파라미터 설명, 모듈 위주의 실행 리뷰)
- Google Colab으로 SKT에서 제공한 모듈을 실행시켰다.
- 전반적인 모델 작동 원리, 각 모듈의 Parameter 및 실행결과를 파악했다.
- 이를 기반으로 코드 내부로 들어가 각 모듈의 인스턴스 등을 분석할 예정.
- https://www.youtube.com/watch?v=S42vDzJExIA

@ DSBA - Multilingual BERT 논문 Review
- Google에서 발표한 BERT의 다국어 버전 논문 Reading
- http://dsba.korea.ac.kr/seminar/?pageid=3&mod=document&keyword=BERT&uid=47

---
#### ~ 21/05/09 ]
@ 210425 Meeting에서 도출된 추후 연구과제 1 수행
 - BERT 모델에 대한 정확한 이해를 위해 논문 Reading
 - https://www.researchgate.net/publication/346870334_Data_Augmentation_Using_Pre-trained_Transformer_Models

@ 210425 Meeting에서 도출된 추후 연구과제 2 수행
 - 논문 git에 있는 CBERT(Baseline)의 구조, 메소드, parameter 등 파악
 - https://github.com/varinf/TransformersDataAugmentation

@ 고려대학교 강필성 교수님 BERT Paper review 영상 시청
 - 논문 reading을 위한 이해 증진 및 놓친 부분 체크
 - https://www.youtube.com/watch?v=IwtexRHoWG0

@ CBERT.py 코드 분석
 - 진행상황 : main()에서 parameters 설정 후 
              train_cbert_and_augment() 메소드 구현중에
              convert_examples_to_features() 메소드 구현(분석)중

 - 다음주 목표 :	1. convert_examples_to_features() 메소드와
                    augment_train_data() 메소드를 중점으로 분석(구현)
                2. 코드 작성 완료하여 CBERT 세부 구조 파악 후 학습
 - https://github.com/varinf/TransformersDataAugmentation/blob/main/src/bert_aug/cbert.py









.
