import tensorflow as tf
import numpy as np

# Get the training dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Split between train and test
# Create model
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## Inference model
y = tf.nn.softmax(tf.matmul(x, W) + b)

## Training model
y_ = tf.placeholder(tf.float32, [None, 10])

# -> Defining cost function
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# Train model
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

sess.run(init)

for _ in range(3000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_:batch_y})

# Test model and report accurary
correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(val)
