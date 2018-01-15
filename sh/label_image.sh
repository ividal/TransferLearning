#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=128

python -m scripts.label_image \
  --model_root_dir="tf_files/models/retrained" \
  --labels=tf_files/models/retrained/retrained_labels.npy \
  --image="tf_files/split_flowers/test/daisy/5547758_eea9edfd54_n.jpg" \
  --input_height=$IMAGE_SIZE \
  --input_width=$IMAGE_SIZE
