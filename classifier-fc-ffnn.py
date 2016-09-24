from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import time
import math
import random
import numpy as np
from PIL import Image
from six.moves import xrange

# Basic model parameters as external flags.
# this is an interesting way of setting up constants
# syntax is flags.define_<type_of_variable>(variable_name, value, description)
flags = tf.app.flags
FLAGS = flags.FLAGS

NUM_CLASSES = 10
IMAGE_SIZE = 28
CHANNELS = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUMBER_OF_TEST_IMAGES=4 #total number of test images

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference(images):

  # Convolutional - 1
  with tf.name_scope('h_conv_1'):
    W_conv_1 = weight_variable([5,5,1,32])
    b_conv_1 = bias_variable([32])

    x_image = tf.reshape(images, [-1,28,28, 1])
    h_conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)

    # Max pool 2x2 - 1
    h_pool_1 = max_pool_2x2(h_conv_1)
  # Convolutional - 2
  with tf.name_scope('h_conv_2'):
    W_conv_2 = weight_variable([5,5,32,64])
    b_conv_2 = bias_variable([64])

    h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)

    # Max pool 2x2 - 1
    h_pool_2 = max_pool_2x2(h_conv_2)

  # Densely connected 1
  with tf.name_scope('densely_connected_1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc1) + b_fc1)

    #keep_prob = tf.constant(0.2, dtype=tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  # Readout Layer
  with tf.name_scope('softmax_linear'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(logits)

  return logits

def cal_loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


# Get the sets of images and labels for training, validation, and
test_images = []
#for filename in ['cats/1.jpg']:
#  image = Image.open(filename)
#  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
#  test_images.append(np.array(image))
for filename in ['mnist_png/testing/0/10.png','mnist_png/testing/1/1004.png','mnist_png/testing/1/1008.png','mnist_png/testing/9/1000.png']:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  test_images.append(np.array(image))

test_images = np.array(test_images)
test_images = test_images.reshape(NUMBER_OF_TEST_IMAGES,IMAGE_PIXELS)

with tf.Graph().as_default():
  images_placeholder, labels_placeholder = placeholder_inputs(NUMBER_OF_TEST_IMAGES)
  logits = inference(images_placeholder)
  loss = cal_loss(logits, labels_placeholder)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  saver.restore(sess, "./data-11999")

  predict_score = norm_score.eval(session = sess,feed_dict={images_placeholder: test_images})
  print("[   0        1      2   3  4 5 6 7  8 9 ]")
  print(predict_score)
