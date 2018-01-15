#!/usr/bin/env bash

# This assumes you have wget installed (`sudo apt-get install wget`).

# Download the pre-trained model (no need to unzip the file):
echo -e "\n--1.\tDownloading the pre-trained model into ./tf_files/models"
mkdir -p $HOME/.keras/models
MODEL=$HOME/.keras/models/mobimobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
if [ ! -f "$MODEL" ]; then
    wget -P $HOME/.keras/models/ https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
fi
echo -e  "--Done.\t--"

# Download the images and unzip them into the correct folders:
echo -e  "\n--2.\tDownloading & extracting the images into ./tf_files/split_flowers"
FLOWERS="tf_files/split_flowers"
VALDIR="$FLOWERS/val"
if [ ! -d "$VALDIR" ]; then
    mkdir -p $FLOWERS && \
    wget -qO- https://www.dropbox.com/s/n257xs7qvnlfik8/split_flowers.tgz?dl=0 | tar xvz -C $FLOWERS --strip-components 1
fi
echo -e  "--Done.\t--"
