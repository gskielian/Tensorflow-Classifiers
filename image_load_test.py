from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import time
import os
import math
import numpy
import numpy as np
import random
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Basic model parameters as external flags.
# this is an interesting way of setting up constants
# syntax is flags.define_<type_of_variable>(variable_name, value, description)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

NUM_CLASSES = 2
IMAGE_SIZE = 28
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
#NUMBER_OF_INPUTS=8 #now determined by direct counting

# Get the sets of images and labels for training, validation, and
train_images = []
train_labels = []


filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./cats/*.jpg"))

image_reader = tf.WholeFileReader()


_, image_file = image_reader.read(filename_queue)

image = tf.image.decode_jpeg(image_file)
resized_image = tf.image.resize_images(image, 28, 28)

with tf.Session() as sess:
    tf.initialize_all_variables().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    image_tensor = sess.run([resized_image])
    print(resized_image)

    coord.request_stop()
    coord.join(threads)
