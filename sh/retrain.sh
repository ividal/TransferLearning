#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=224

# 1.0, 0.75, 0.50 or 0.25
WIDTH=1.0

# adam or sgd
OPTIMIZER="sgd"
LEARNING_RATE=0.01
BATCH_SIZE=8
STEPS=500
LABEL="bs_$BATCH_SIZE-lr_$LEARNING_RATE-opt_$OPTIMIZER"

python -m scripts.retrain \
  --model_dir=tf_files/models \
  --summaries_dir=tf_files/training_summaries/$LABEL \
  --output_graph="tf_files/retrained_graph_${WIDTH}.pb" \
  --image_dir=tf_files/split_flowers \
  --how_many_training_steps=$STEPS \
  --learning_rate=$LEARNING_RATE\
  --train_batch_size=$BATCH_SIZE \
  --optimizer_name=$OPTIMIZER \
  --image_size=$IMAGE_SIZE
