#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import DATA_DIR
from keras.preprocessing.image import img_to_array, load_img
from os.path import join
from pandas import read_csv


def read_img(img_id, train_or_test, size):
    if train_or_test not in ['train', 'test']:
        raise ValueError('wrong value for train_or_test!')
    path = join(DATA_DIR, train_or_test, f'{img_id}.jpg')
    img = load_img(path, target_size=size)
    return img_to_array(img)


def get_labels():
    path = join(DATA_DIR, 'labels.csv')
    return read_csv(path)
