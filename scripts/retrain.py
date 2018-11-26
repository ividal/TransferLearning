import argparse
from glob import glob
import os
import sys
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet

tf.logging.set_verbosity(tf.logging.INFO)


def create_model(num_labels, input_shape, learning_rate=1e-3, fully_trainable=False):
    print("\tCreating an MobileNet model with Imagenet weights. "
          "Classes: {}".format(num_labels))
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape,
                             input_tensor=tf.keras.Input(input_shape), pooling='avg')

    pred_name = "predictions_{}".format(num_labels)

    x = base_model.output
    predictions = tf.keras.layers.Dense(num_labels, activation='softmax', name=pred_name)(x)

    # create graph of your new model
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    for l in model.layers:
        l.trainable = fully_trainable

    classifier = model.get_layer(pred_name)
    classifier.trainable = True

    print(model.summary())

    print("\tlast layer: {}".format(model.layers[-1].output_shape))

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

    # We will now compile and print out a summary of our model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def create_generators(train_dir, val_dir, test_dir, batch_size, image_size=224):
    #preprocess_input = tf.keras.applications.mobilenet.preprocess_input

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # ,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical')

    return train_gen, val_gen, test_gen


def get_folder_info(image_dir):
    categories = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    num_categories = len(categories)
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(image_dir, "**/*.{}".format(extension))
        file_list.extend(glob(file_glob))

    num_images = len(file_list)

    return num_images, num_categories


def create_callbacks(output_model_path, summary_dir):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_model_path, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=summary_dir, write_graph=True, write_images=True, histogram_freq=1)
    ]
    return callbacks


def main(_):
    train_dir = os.path.join(FLAGS.image_dir, "train")
    val_dir = os.path.join(FLAGS.image_dir, "val")
    test_dir = os.path.join(FLAGS.image_dir, "test")
    retrained_model_path = os.path.join(FLAGS.model_dir, "retrained.h5")

    num_train_images, num_labels = get_folder_info(train_dir)

    model = create_model(input_shape=(FLAGS.image_size, FLAGS.image_size, 3),
                         num_labels=num_labels,
                         learning_rate=FLAGS.learning_rate)

    callbacks = create_callbacks(output_model_path=retrained_model_path, summary_dir=FLAGS.summaries_dir)

    train_gen, val_gen, test_gen = create_generators(train_dir, val_dir, test_dir, batch_size=FLAGS.train_batch_size)

    untrained_path = tf.contrib.saved_model.save_keras_model(model, FLAGS.model_dir)
    untrained = tf.contrib.saved_model.load_keras_model(untrained_path)
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # We will now compile and print out a summary of our model
    untrained.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('Testing untrained model:')
    _, untrained_accuracy = untrained.evaluate_generator(test_gen)
    print('Untrained accuracy: {}'.format(untrained_accuracy))


    model.fit_generator(
        train_gen,
        steps_per_epoch=num_train_images // FLAGS.train_batch_size,
        epochs=1,
        validation_data=val_gen,
        validation_steps=10,
        callbacks=callbacks)

    print('Testing retrained model:')
    loss, test_accuracy = model.evaluate_generator(test_gen)
    print("Model's test accuracy: {}".format(test_accuracy))

    output_path = tf.contrib.saved_model.save_keras_model(model, FLAGS.model_dir)
    loaded_model = tf.contrib.saved_model.load_keras_model(output_path)
    optimizer = tf.keras.optimizers.SGD()

    # We will now compile and print out a summary of our model
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('Testing loaded model:')
    _, loaded_accuracy = loaded_model.evaluate_generator(test_gen)
    print('Test accuracy: {}'.format(loaded_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help="Path to folders of labeled images."
    )
    parser.add_argument(
        "--output_graph",
        type=str,
        default="/tmp/output_graph.pb",
        help="Where to save the trained graph."
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
        "--testing_percentage",
        type=int,
        default=10,
        help="What percentage of images to use as a test set."
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=10,
        help="What percentage of images to use as a validation set."
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int,
        default=10,
        help="How often to evaluate the training results."
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
        default="/tmp/imagenet",
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        "--bottleneck_dir",
        type=str,
        default="/tmp/bottleneck",
        help="Path to cache bottleneck layer values as files."
    )
    parser.add_argument(
        "--final_tensor_name",
        type=str,
        default="final_result",
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenet_1.0_224",
        help="""\
      Which model architecture to use. For faster or smaller models, choose a MobileNet with the 
      form 'mobilenet_<parameter size>_<input_size>'. For example, 'mobilenet_1.0_224' will pick 
      a model that is 17 MB in size and takes 224 pixel input images.\
      """)
    parser.add_argument(
        "--max_num_images_per_class",
        type=int,
        default=2 ** 27 - 1,
        help="""\
      The maximum number of images to be allowed for a single class. The default number is huge 
      even for COCO or ImageNet standards (2**27-1).\
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
