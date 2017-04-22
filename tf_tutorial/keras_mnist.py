import keras as k
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.datasets import mnist


# Data set and preprocessing
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test  = X_test.reshape(X_test.shape[0], 1, 28 ,28)

X_train = X_train.astype('float32')
X_train /= 255
X_test  = X_test.astype('float32')
X_test /= 255

# Smaller sized dataset to check how tensorboard works
x_train = X_train[:1000, :]
y_train = Y_train[:1000]

Y_train = np_utils.to_categorical(Y_train, 10)
y_train = np_utils.to_categorical(y_train, 10)
Y_test  = np_utils.to_categorical(Y_test, 10)

#Define the model here
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

#Stitch all the layers together
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
history = model.fit(x_train, y_train, batch_size=100, nb_epoch=100, verbose=1,
                    callbacks=[TensorBoard(log_dir='./tensorboard_log', histogram_freq=1)])
print(history)
#See the output on test data
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)

