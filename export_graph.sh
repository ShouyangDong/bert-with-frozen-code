export MODEL_CKPT_DIR=./chinese_L-12_H-768_A-12
export MODEL_PB_DIR = ./pbfile_dir
export XNLI_DIR=./xnli

python3.5 ./bert/export_inference_graph_bert.py \
  --model_dir=$MODEL_CKPT_DIR
  --task_name=XNLI \
  --data_dir=$XNLI_DIR \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --ckpt_file=$MODEL_CKPT_DIR/bert_model.ckpt \
  --serving_model_save_path = $MODEL_PB_DIR
