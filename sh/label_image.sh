#!/usr/bin/env bash

# 224, 192, 160 or 128
IMAGE_SIZE=224

# 1.0, 0.75, 0.50 or 0.25
WIDTH=1.0

ARCHITECTURE="mobilenet_v1_${WIDTH}_${IMAGE_SIZE}"

python -m scripts.label_image \
  --model_file="tf_files/retrained_graph_${WIDTH}.pb" \
  --labels=tf_files/retrained_labels.txt \
  --image="tf_files/flower_photos/daisy/3475870145_685a19116d.jpg" \
  --input_height=$IMAGE_SIZE \
  --input_width=$IMAGE_SIZE
