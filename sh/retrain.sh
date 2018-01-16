#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=224

# 1.0, 0.75, 0.50 or 0.25
WIDTH=1.0

# adam or sgd
OPTIMIZER="sgd"
LEARNING_RATE=0.01
BATCH_SIZE=128
TEST_PERC=5
STEPS=500
LABEL="bs_$BATCH_SIZE-lr_$LEARNING_RATE-opt_$OPTIMIZER"

ARCHITECTURE="mobilenet_v1_${WIDTH}_${IMAGE_SIZE}"

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models \
  --summaries_dir=tf_files/training_summaries/$LABEL \
  --output_graph="tf_files/retrained_graph_${WIDTH}.pb" \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --testing_percentage=$TEST_PERC \
  --how_many_training_steps=$STEPS \
  --learning_rate=$LEARNING_RATE\
  --train_batch_size=$BATCH_SIZE \
  --optimizer_name=$OPTIMIZER
