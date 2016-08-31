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
# Get the sets of images and labels for training, validation, and
train_images = []
train_labels = []

image_reader = tf.WholeFileReader()

"""start get image paths"""

def get_image_paths_in_folder(folder_name):
  image_paths = [os.path.join(folder, pic)
      for folder, subs, pics, in os.walk(".")
      for pic in pics if pic.endswith(".jpg") and folder.startswith(folder_name)]
  return image_paths

cat_image_paths = get_image_paths_in_folder("./cats")
dog_image_paths = get_image_paths_in_folder("./dogs")

for filename in cat_image_paths:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  train_images.append(np.array(image))
for filename in dog_image_paths:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  train_images.append(np.array(image))

NUMBER_OF_CAT_IMAGES = len(cat_image_paths)
NUMBER_OF_DOG_IMAGES = len(dog_image_paths)

"""end get image paths"""
all_image_paths = cat_image_paths + dog_image_paths
#print(all_image_paths)

filename_queue = tf.train.string_input_producer(all_image_paths, num_epochs = None)

#print(filename_queue)

print(image_reader.read(filename_queue))

_, image_file = image_reader.read(filename_queue)

images = tf.image.decode_jpeg(image_file)


#this is now done beforehand with the imagemagick convert script
#resized_image = tf.image.resize_images(image, 28, 28) 


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    image_tensor = []
    for i in range(2):
        image_tensor.append(sess.run(images))

    #print individual images
    print(image_tensor[0])
    time.sleep(1)
    print(image_tensor[1])

    #print(image_tensor)
    coord.request_stop()
    coord.join(threads)
