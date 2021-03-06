# Copyright 2018 Irina Vidal Migallon. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse

import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model import load_keras_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.logging.set_verbosity(tf.logging.INFO)


def load_labels(label_file):
    trained_classes = np.load(label_file).item()
    tf.logging.info("Loaded the following class map: {}".format(label_file))
    return trained_classes


def classify(args):
    image = img_to_array(
        load_img(args.image_path, target_size=(args.input_height, args.input_width)))
    image = image.reshape([1, args.input_height, args.input_width, -1])
    image = preprocess_input(image)

    tf.logging.info("Loading model: {}".format(args.model_root_dir))
    model = load_keras_model(args.model_root_dir)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    results = model.predict(image)

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    tf.logging.info(top_k)
    labels = load_labels(args.labels)
    tf.logging.debug(labels)

    for i in top_k:
        print(labels[i], results[i])


if __name__ == "__main__":
    """
    The expected way and paths used to call this script is as follows:
    python -m scripts.label_image \
      --model_root_dir="tf_files/models/retrained" \
      --labels=tf_files/models/retrained/retrained_labels.npy \
      --image="tf_files/split_flowers/test/daisy/5547758_eea9edfd54_n.jpg"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="image to be processed", required=False)
    parser.add_argument("--model_root_dir", type=str, help="directory where the SavedModel "
                                                           "files are", required=True)
    parser.add_argument("--labels", type=str, help="name of file containing labels", required=False)
    parser.add_argument("--input_height", type=int, help="input height", default=224)
    parser.add_argument("--input_width", type=int, help="input width", default=224)
    args = parser.parse_args()

    classify(args)
