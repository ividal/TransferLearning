#!/usr/bin/env bash

# This assumes you have wget installed (`sudo apt-get install wget`).

# Download the pre-trained model (no need to unzip the file):
echo -e "\n--1.\tDownloading the pre-trained model into ./tf_files/models"
wget -P tf_files/models/ https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
echo -e  "--Done.\t--"

# Download the images and unzip them into the correct folders:
echo -e  "\n--2.\tDownloading & extracting the images into ./tf_files/flower_photos"
wget -qO- http://download.tensorflow.org/example_images/flower_photos.tgz | tar --one-top-level=tf_files -xvz
echo -e  "--Done.\t--"
