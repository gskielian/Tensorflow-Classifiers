import tensorflow as tf

import time
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
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 2, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
NUM_CLASSES = 2
IMAGE_SIZE = 28
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
NUMBER_OF_INPUTS=2 #total number of images
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

def training(loss, learning_rate):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(images_feed,labels_feed, images_pl, labels_pl):
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = 8 // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(train_images,test_labels,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

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

#label = [1,0]
label = [1]
test_labels = np.array(label)
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
  print predict_score
