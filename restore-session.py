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
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')

NUM_CLASSES = 2
IMAGE_SIZE = 28
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUMBER_OF_TEST_IMAGES=2 #total number of test images

def inference(images, hidden1_units, hidden2_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
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
for filename in ['cats/1.jpg','cats/2.jpg']:
  image = Image.open(filename)
  image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
  test_images.append(np.array(image))

test_images = np.array(test_images)
test_images = test_images.reshape(NUMBER_OF_TEST_IMAGES,IMAGE_PIXELS)

with tf.Graph().as_default():
  images_placeholder, labels_placeholder = placeholder_inputs(NUMBER_OF_TEST_IMAGES)
  logits = inference(images_placeholder,
      FLAGS.hidden1,
      FLAGS.hidden2)
  loss = cal_loss(logits, labels_placeholder)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  saver.restore(sess, "./data-1999")

  predict_score = norm_score.eval(session = sess,feed_dict={images_placeholder: test_images})
  print("[   dog score        cat score     ]")
  print predict_score
