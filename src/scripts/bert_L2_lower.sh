#!/usr/bin/env bash

SRC=/home/delabgpu/Tae_StudyCode/TransformersDataAugmentation/src
CACHE=/home/delabgpu/Tae_StudyCode/TransformersDataAugmentation/CACHE
TASK=trec

for NUMEXAMPLES in 10;
do
    for i in {0..14};
        do
        RAWDATADIR=/home/delabgpu/Tae_StudyCode/TransformersDataAugmentation/src/utils/datasets/L2/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE > $RAWDATADIR/bert_baseline.log

      ##############
      ## EDA
      ##############

      EDADIR=$RAWDATADIR/eda
      mkdir $EDADIR
      python $SRC/bert_aug/eda.py --input $RAWDATADIR/train.tsv --output $EDADIR/eda_aug.tsv --num_aug=1 --alpha=0.1 --seed ${i}
      cat $RAWDATADIR/train.tsv $EDADIR/eda_aug.tsv > $EDADIR/train.tsv
      cp $RAWDATADIR/test.tsv $EDADIR/test.tsv
      cp $RAWDATADIR/dev.tsv $EDADIR/dev.tsv
      python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $EDADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE  > $RAWDATADIR/bert_eda.log


    #    #######################
    #    # Backtranslation DA Classifier
    #    #######################

    BTDIR=$RAWDATADIR/bt
    mkdir $BTDIR
    python $SRC/bert_aug/backtranslation.py --data_dir $RAWDATADIR --output_dir $BTDIR --task_name $TASK  --seed ${i} --cache $CACHE
    cat $RAWDATADIR/train.tsv $BTDIR/bt_aug.tsv > $BTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $BTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $BTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $BTDIR --seed ${i} --cache $CACHE  > $RAWDATADIR/bert_bt.log



   # #######################
   # # CMODBERTP Classifier
   # ######################

    CMODBERTPDIR=$RAWDATADIR/cmodbertp
    mkdir $CMODBERTPDIR
    python $SRC/bert_aug/cmodbertp.py --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log
    cat $RAWDATADIR/train.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE > $RAWDATADIR/bert_cmodbertp.log


    done
done


