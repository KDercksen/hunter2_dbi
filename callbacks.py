#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.callbacks import Callback
from time import time


class MaxTrainingMinutes(Callback):

    def __init__(self, minutes, verbose=0):
        super(MaxTrainingMinutes, self).__init__()
        self.minutes = minutes
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.initiated = time()
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = time()
        if current - self.minutes * 60 >= self.initiated:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1}: time up, stopping')
