#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=128

# 1.0, 0.75, 0.50 or 0.25
WIDTH=1.0

# adam, sgd or default
OPTIMIZER="adam"
LEARNING_RATE=0.001
BATCH_SIZE=8
STEPS=800
ARCH="mobilenet"
LABEL="$ARCH-bs_$BATCH_SIZE-lr_$LEARNING_RATE-opt_$OPTIMIZER"

python -m scripts.retrain \
  --model_dir=tf_files/models \
  --bottleneck_dir=tf_files/bottlenecks \
  --summaries_dir=tf_files/training_summaries/$LABEL \
  --output_graph="tf_files/retrained_graph_${WIDTH}.pb" \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=tf_files/split_flowers \
  --how_many_training_steps=$STEPS \
  --learning_rate=$LEARNING_RATE\
  --train_batch_size=$BATCH_SIZE \
  --optimizer_name=$OPTIMIZER \
  --image_size=$IMAGE_SIZE
