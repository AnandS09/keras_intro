import numpy as np
import tensorflow as tf
import random

# Training data
# x_train = np.linspace(0, 50, 50)
# y_train = 2 * x_train + 3 + random.random()  #Add some random perturbations

# x_train = list(range(0, 50))
# y_train = x_train
# for i in range(len(x_train)):
#     y_train[i] = 2 * x_train[i] + random.random()

x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

alpha = tf.constant(0.01, tf.float32)

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

lin_model = W * x + b

loss = tf.reduce_sum(tf.square(lin_model - y))

optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("Iteration " + str(i) + " W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))


