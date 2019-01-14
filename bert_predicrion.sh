#!/bin/bash

export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export FROZEN_MODEL_DIR=./frozenAPI
export XNLI_DIR=./xnli

python3.5 ./bert/bert_api.py \
  --task_name=XNLI \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --model_path=$FROZEN_MODEL_DIR \
  --max_seq_length=128 \