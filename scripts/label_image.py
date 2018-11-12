# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    session = tf.Session()
    result = session.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    """
    The expected way and paths used to call this script is as follows:
    python -m scripts.label_image \
      --model_file="tf_files/retrained_graph_${WIDTH}.pb" \
      --labels=tf_files/retrained_labels.txt \
      --image="tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="image to be processed", required=False)
    parser.add_argument("--model_file", type=str, help="graph/model to be executed", required=True)
    parser.add_argument("--labels", type=str, help="name of file containing labels", required=False)
    parser.add_argument("--input_height", type=int, help="input height", default=224)
    parser.add_argument("--input_width", type=int, help="input width", default=224)
    parser.add_argument("--input_mean", type=int, help="input mean", default=128)
    parser.add_argument("--input_std", type=int, help="input std", default=128)
    parser.add_argument("--input_layer", type=str, help="name of input layer", default="input")
    parser.add_argument("--output_layer", type=str, help="name of output layer", default="final_result")
    args = parser.parse_args()

    if args.labels:
        label_file = args.labels

    graph = load_graph(args.model_file)
    t = read_tensor_from_image_file(args.image_path,
                                    input_height=args.input_height,
                                    input_width=args.input_width,
                                    input_mean=args.input_mean,
                                    input_std=args.input_std)

    input_name = "import/" + args.input_layer
    output_name = "import/" + args.output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.labels)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))

    for i in top_k:
        print(labels[i], results[i])
