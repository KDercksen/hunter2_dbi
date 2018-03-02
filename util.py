#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import DATA_DIR
from keras.preprocessing.image import img_to_array, load_img
from pandas import read_csv
import numpy as np
import os
import os.path as osp


def read_img(img_id, train_or_test, size):
    '''Read a single image into NumPy array.

    Arguments:
    ----------
    img_id: string
        Image ID. For example if image filename is xxx.jpg, 'xxx' is the ID.
    train_or_test: string
        'train' or 'test'.
    size: int
        Size of image (size x size), for example 224 for VGG16 network or 299
        for XCeption network.

    Returns:
    --------
    img_array: numpy array
        Image loaded as NumPy array.
    '''
    if train_or_test not in ['train', 'test']:
        raise ValueError('wrong value for train_or_test!')
    path = osp.join(DATA_DIR, train_or_test, f'{img_id}.jpg')
    img = load_img(path, target_size=(size, size))
    return img_to_array(img)


def get_labels():
    '''Get training labels.

    Returns:
    --------
    labels: pandas dataframe
        Dataframe has two columns, 'id' and 'breed'. 'id' is the image id,
        'breed' is the classlabel for that image.
    '''
    path = osp.join(DATA_DIR, 'labels.csv')
    return read_csv(path)


def get_images(train_or_test, size):
    '''Generator that yields images from train or test set.

    Arguments:
    ----------
    train_or_test: string
        'train' or 'test'.
    size: int
        Size of image (size x size), for example 224 for VGG16 network or 299
        for XCeption network.

    Yields:
    -------
        img_array: numpy array
            Image loaded as NumPy array.
    '''
    for img in os.listdir(osp.join(DATA_DIR, train_or_test)):
        img_id = osp.splitext(osp.basename(img))[0]
        yield read_img(img_id, train_or_test, size), img_id


def basic_images_generator(train_or_test, labels, batch_size, size):
    '''Generator for use with Keras' model.fit_generator.

    TODO: Make this use pivoted labels somehow?

    Arguments:
    ----------
    train_or_test: string
        'train' or 'test'.
    labels: pandas dataframe
        A pandas dataframe with columns 'id' and 'breed'. For this you can use
        the output (or some subset) of util.get_labels.
    batch_size: int
        Number of samples and labels to return for each batch.
    size:
        Size of image (size x size), for example 224 for VGG16 network or 299
        for XCeption network.

    Yields:
    -------
    batch_samples, batch_labels: tuple of np.array
        A tuple of samples and corresponding labels.
    '''
    batch_samples = np.zeros((batch_size, size, size, 3))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            idx = np.random.choice(len(labels), 1)
            batch_samples[i] = read_img(labels.iloc[idx]['id'], train_or_test,
                                        size)
            batch_labels[i] = labels.iloc[idx]['breed']
        yield batch_samples, batch_labels
