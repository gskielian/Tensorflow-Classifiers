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



def get_image_paths_in_folder(folder_name):
  image_paths = [os.path.join(folder, pic)
      for folder, subs, pics, in os.walk(".")
      for pic in pics if pic.endswith(".jpg") and folder.startswith(folder_name)]
  return image_paths

cat_image_paths = get_image_paths_in_folder("./cats")
dog_image_paths = get_image_paths_in_folder("./dogs")
#print cat_image_paths
#print dog_image_paths

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



label = [1]*NUMBER_OF_CAT_IMAGES + [0]*NUMBER_OF_DOG_IMAGES
print(np.array(label))
train_labels = tf.pack(np.array(label))

print(train_labels.get_shape()[0])

NUMBER_OF_INPUTS = NUMBER_OF_CAT_IMAGES + NUMBER_OF_DOG_IMAGES

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
    self.last_index = train_labels.get_shape()[0] - 1

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

print(data_set.batch_size)


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

def fill_feed_dict(data_set, images_pl, labels_pl):

  images_feed, labels_feed = data_set.next_batch()

  images_feed_tensor = tf.convert_to_tensor(images_feed, dtype=tf.float32)
  labels_feed_tensor = tf.convert_to_tensor(labels_feed, dtype=tf.int32)
  print("images shape is: ")
  print(images_feed_tensor)
  print("labels shape is: ")
  print(labels_feed_tensor)

  feed_dict = {
      images_pl: images_feed_tensor,
      labels_pl: labels_feed_tensor,
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
    logits = inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

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
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                train_images)


def main(_):
  run_training()
if __name__ == '__main__':
  tf.app.run()
