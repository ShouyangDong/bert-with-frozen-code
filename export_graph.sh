export MODEL_CKPT_DIR=./xnli_output_gpu
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export MODEL_PB_DIR=./bertAPI
export XNLI_DIR=./xnli

python3.5 ./bert/export_inference_graph_bert.py \
  --model_dir=$MODEL_CKPT_DIR \
  --task_name=XNLI \
  --data_dir=$XNLI_DIR \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --ckpt_file=$MODEL_CKPT_DIR/model.ckpt-32725 \
  --serving_model_save_path=$MODEL_PB_DIR
