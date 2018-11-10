from __future__ import print_function

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.layers import *
import keras.layers
from keras.models import *
from keras.optimizers import RMSprop

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


batch_size = 128
num_classes = 10
epochs = 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class NnModel:
    def __init__(self, lr):
        print("Initializing model with LR =", lr)
        self.lr = lr
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(784,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=lr),
                      metrics=['accuracy'])

    def evaluate(self):
        history = self.model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_test, y_test))

        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return score[1]

