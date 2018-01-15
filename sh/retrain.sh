#!/usr/bin/env bash

export IMAGE_SIZE=224
export WIDTH=1.0
export ARCHITECTURE="mobilenet_v1_${WIDTH}_${IMAGE_SIZE}"
export PYTHONPATH=".."

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --how_many_training_steps=500 \
  --learning_rate=0.01\
  --testing_percentage=10 \
  --validation_percentage=10