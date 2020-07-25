# -*- encoding utf8 -*-

import tensorflow.compat.v1 as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


class QNetwork():

    def __init__(self, statesize, actionsize):
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=statesize))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(actionsize, activation='linear'))
        self.model.compile(optimizer=Adam(
            learning_rate=0.001), loss=tf.losses.huber_loss)


if __name__ == "__main__":
    statesize = 4
    actionsize = 2
    network = QNetwork(statesize, actionsize)
