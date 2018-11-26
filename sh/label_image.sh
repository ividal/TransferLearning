#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=128

python -m scripts.label_image \
  --model_root_dir="tf_files/models/1543270820" \
  --labels=tf_files/retrained_labels.npy \
  --image="tf_files/flower_photos/daisy/3475870145_685a19116d.jpg" \
  --input_height=$IMAGE_SIZE \
  --input_width=$IMAGE_SIZE
