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
from glob import glob
import logging
import numpy as np
import os
from distutils.dir_util import copy_tree
import sys
import time
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("training")


def create_model(num_labels, input_shape, learning_rate=1e-3, optimizer=None, fully_trainable=False):
    """
    Builds the neural network and compiles the model with the desired optimizer.
    :param num_labels: total number of categories of our multiclass classifier
    :param input_shape: the 2D (h, w) of our input
    :param learning_rate: the learning rate for our optimizer
    :param optimizer: either a tf.keras optimizer or a tf.train.optimizer
    :param fully_trainable: whether we want the feature extractor layers to be trainable or not
    :return: A compiled model ready to be trained or used for inference.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def create_feature_extractor(input_shape, fully_trainable=False):
    """
    Builds a NN model that takes in an image and produces bottleneck descriptors, i.e. it does not have a
    classification head. Preferably based on a model with pre-trained weights.
    :param input_shape: 2D (h, w) of the input.
    :param fully_trainable: Whether the layers should be trainable or not
    :return: a tf.keras model, not yet compiled
    """
    logger.info("\tCreating an MobileNet model with Imagenet weights. ")
    base_model = None

    return base_model


def add_classifier(base_model, num_labels):
    """
    Adds the final classifying top to our feature extraction model.
    :param base_model: the feature extractor
    :param num_labels: the number of classes of this multiclass classifier
    :return: a classification architecture based on the passed feature extractor
    """
    logger.info("\tAdding a {} label classifier.".format(num_labels))

    model = None
    return model



def create_data_feeders(train_dir, val_dir, test_dir, batch_size, image_size=224):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, preprocessing_function=preprocess_input)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical")

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical")

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical")

    return train_gen, val_gen, test_gen


def create_callbacks(output_model_path, summary_dir):
    pass


def evaluate_model(path, image_gen):
    pass


def save_labels_to_file(train_gen, labels_file):
    """
    Saves the relationship between class names to its one-hot encoding
    :param train_gen: Training data generator
    :param labels_file: output labels file path
    """
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    trained_classes = train_gen.class_indices
    classes_by_idx = {v: k for k, v in trained_classes.items()}
    logger.info("Saving trained classes to {}".format(labels_file))
    np.save(labels_file, classes_by_idx)


def main(_):
    train_dir = os.path.join(FLAGS.image_dir, "train")
    val_dir = os.path.join(FLAGS.image_dir, "val")
    test_dir = os.path.join(FLAGS.image_dir, "test")
    checkpoint_dir = os.path.join(os.path.dirname(FLAGS.model_dir), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pb")
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join(FLAGS.summaries_dir, timestamp)

    logger.info("\n===\tSetting up data loaders for train, val and test data.")
    train_gen, val_gen, test_gen = create_data_feeders(train_dir, val_dir, test_dir,
                                                       batch_size=FLAGS.train_batch_size,
                                                       image_size=FLAGS.image_size)

    logger.info("\n===\tSaving key file with label names <> index conversion.")

    save_labels_to_file(train_gen, FLAGS.output_labels)

    logger.info("\n===\tCreating a classification model based on pre-trained weights.")

    logger.info("\n===\tPreparing Tensorboard callback, to monitor training.")

    logger.info("\n===\tRetraining downloaded model.")

    logger.info("\n===\tExporting the model so it can be served (TF Serving, TF Lite, etc.).")

    logger.info("\n===\tReporting final model accuracy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Path to folders of labeled images."
    )
    parser.add_argument(
        "--output_labels",
        type=str,
        default="/tmp/output_labels.txt",
        help="Where to save the trained graph's labels."
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="/tmp/retrain_logs",
        help="Where to save summary logs for TensorBoard."
    )
    parser.add_argument(
        "--how_many_training_steps",
        type=int,
        default=200,
        help="How many training steps to run before ending."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="How large a learning rate to use when training."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=100,
        help="How many images to train on at a time."
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=100,
        help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="tf_files/models/retrained",
        help="""\
      Path to store the exported tf.keras model in a format that tensorflow can cleanly import and serve. It must be 
      a directory, since SavedModel has its own internal structure.\
      """
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenetv2_1.0_224",
        help="""\
      Which model architecture to use. For faster or smaller models, choose a MobileNet with the 
      form 'mobilenet_<parameter size>_<input_size>'. For example, 'mobilenet_1.0_224' will pick 
      a model that is 17 MB in size and takes 224 pixel input images.\
      """)
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="sgd",
        help="""\
      The optimizer to use. Accepted values are currently 'adam' or 'sgd'\
      """)
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="""\
      The image width (and height, they're square) to use. Accepted values are currently 224, 160 and 128.\
      """)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="""\
      Also used to pace training and affect the visualization of the training. If used together \
      with the number of steps, steps_per_epoch will be tweaked.\
      """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
