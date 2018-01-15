#!/usr/bin/env bash

IMAGE_SIZE=224
WIDTH=1.0
ARCHITECTURE="mobilenet_v1_${WIDTH}_${IMAGE_SIZE}"

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph="tf_files/retrained_graph_${WIDTH}.pb" \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --validation_percentage=10 \
  --testing_percentage=1 \
  --how_many_training_steps=100 \
  --learning_rate=0.01\
  --train_batch_size=100 \
  --optimizer_name="sgd"
