# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Irina Vidal Migallon
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

import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.python.framework import graph_util



def create_image_lists(image_dir, testing_percentage, validation_percentage, max_num_images_per_class):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (max_num_images_per_class + 1)) *
                               (100.0 / max_num_images_per_class))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """"Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
      architecture: The name of the model architecture.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + architecture + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

    Args:
      sess: Current active TensorFlow Session.
      image_data: String of raw JPEG data.
      image_data_tensor: Input data layer in the graph.
      decoded_image_tensor: Output of initial image resizing and preprocessing.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: Layer before the final softmax.

    Returns:
      Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract(data_url, dest_directory):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
      data_url: Web location of the tar file containing the pretrained model.
      dest_directory: Folder where the pretrained model will be stored.
    """
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    model_name = os.path.splitext(filename)[0].strip("_frozen")
    model_folder = os.path.join(dest_directory, model_name)
    model_full_path = os.path.join(model_folder, "frozen_graph.pb")

    tf.logging.info("Expecting:\n\t{}\nor\t{}".format(model_folder, filepath))

    ensure_dir_exists(dest_directory)
    ensure_dir_exists(model_folder)

    if not os.path.exists(model_full_path):
        if not os.path.exists(filepath):
            tf.logging.info('Tarfile {} missing, downloading it.'.format(filepath))

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename,
                                  float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            tf.logging.info('Successfully downloaded {}, {} bytes'.format(filename, statinfo.st_size))

        tf.logging.info('Extracting file from {}'.format(filepath))
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        assert os.path.exists(model_full_path), "Unable to extract model: {}".format(
            model_full_path)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        tf.logging.info('Folder {} missing, creating it.'.format(dir_name))
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be modulo-ed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of which set to pull images from - training, testing,
      or validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: The tensor to feed loaded jpeg data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The output tensor for the bottleneck values.
      architecture: The name of the model architecture.

    Returns:
      Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_values = None
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values



def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture, max_num_images_per_class):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: If positive, a random sample of this size will be chosen.
      If negative, all bottlenecks will be retrieved.
      category: Name string of which set to pull from - training, testing, or
      validation.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      image_dir: Root folder string of the subfolders containing the training
      images.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
      architecture: The name of the model architecture.

    Returns:
      List of bottleneck arrays, their corresponding ground truths, and the
      relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(max_num_images_per_class + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor, architecture)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.

    Args:
      sess: The current active TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      image_dir: Root folder string of the subfolders containing the training
      images.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      jpeg_data_tensor: Input tensor for jpeg data from file.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The penultimate output layer of the graph.
      architecture: The name of the model architecture.

    Returns:
      Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')



def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph.

    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])

    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def prepare_file_system(summaries_dir):
    # Setup the directory we'll write summaries to for TensorBoard
    ensure_dir_exists(summaries_dir)
    return


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.

    Args:
      architecture: Name of a model architecture.

    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized

    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[2]
        if (version_string != '1.0' and version_string != '0.75' and
                version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
        but found '%s' for architecture '%s'""",
                version_string, architecture)
            return None
        size_string = parts[3]
        if (size_string != '224' and size_string != '192' and
                size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
       but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        resized_input_tensor_name = 'input:0'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_base_name = 'frozen_graph.pb'

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }
