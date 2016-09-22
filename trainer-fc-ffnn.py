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
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_checkpoint', 'data', 'checkpoint saver prefix')

NUM_CLASSES = 10
IMAGE_SIZE = 28
CHANNELS = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
#NUMBER_OF_INPUTS=8 #now determined by direct counting

# Get the sets of images and labels for training, validation, and
train_images = []
train_labels = []

training_image_paths = []
training_image_numbers = []
labels = []

image_reader = tf.WholeFileReader()

"""start get image paths"""

def get_image_paths_in_folder(folder_name):
  image_paths = [os.path.join(folder, pic)
      for folder, subs, pics, in os.walk(".")
      for pic in pics if pic.endswith(".png") and folder.startswith(folder_name)]
  return image_paths

for x in xrange(NUM_CLASSES):
  training_image_paths.append(get_image_paths_in_folder("./mnist_png/testing/" + str(x)))


#append all images to a single array
for x in xrange(NUM_CLASSES):
  for filename in training_image_paths[x]:
    image = Image.open(filename)
    image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
    train_images.append(np.array(image))

#get the lengths of each into an array
for x in xrange(NUM_CLASSES):
  training_image_numbers.append(len(training_image_paths[x]))

"""end get image paths"""

#start getting labels in format

for x in xrange(NUM_CLASSES):
  labels += [x]*training_image_numbers[x]

train_labels = np.array(labels)


NUMBER_OF_INPUTS = len(train_images)

train_images = np.array(train_images)
train_images = train_images.reshape(NUMBER_OF_INPUTS,IMAGE_PIXELS)

#dataset class for feeding next batch
class DataSet(object):
  train_images = []
  train_labels = []
  batch_index = 0
  batch_size = 1
  last_index = 0

  def __init__(self, train_images, train_labels, batch_size):
    self.train_images = train_images
    self.train_labels = train_labels
    self.batch_size = batch_size
    self.batch_index = 0
    self.last_index = len(train_labels) - 1

  def next_batch(self):
    temp_batch_index = self.batch_index
    start_index = temp_batch_index * self.batch_size
    #end index is one batch ahead of start
    end_index = (temp_batch_index + 1 ) * self.batch_size

    # if within bounds, send over the labels and images
    if end_index <= self.last_index:
      self.batch_index += 1
      return self.train_images[start_index:end_index], self.train_labels[start_index:end_index]
    # else if out of bounds, then wrap around
    # TODO is this where we mark an epoch?
    else:
      self.batch_index = 0
      temp_batch_index = self.batch_index
      start_index = self.batch_index * self.batch_size
      end_index = (self.batch_index + 1 ) * self.batch_size
      return self.train_images[start_index:end_index], self.train_labels[start_index:end_index]

#create data_set object similar to way in which MNIST example was created
data_set = DataSet(train_images, train_labels, FLAGS.batch_size)


test_images, test_labels = data_set.next_batch()

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
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss, learning_rate):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=[batch_size,IMAGE_PIXELS])
  labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):

  images_feed, labels_feed = data_set.next_batch()

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed
  }
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder, labels_placeholder,
            data_set):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = NUMBER_OF_INPUTS // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @1.00: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder)

    # Add to the Graph the Ops for loss calculation.
    loss = cal_loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
      #feed_dict = fill_feed_dict(train_images,train_labels,
      #                           images_placeholder,
      #                           labels_placeholder)
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
      duration = time.time() - start_time
      if step % 2 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_checkpoint, global_step=step)
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_set)

def main(_):
  run_training()
if __name__ == '__main__':
  tf.app.run()
