#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import DATA_DIR
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
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


def get_images(train_or_test, size, img_ids=None, amount=None):
    '''Generator that yields images from train or test set.

    Arguments:
    ----------
    train_or_test: string
        'train' or 'test'.
    size: int
        Size of image (size x size), for example 224 for VGG16 network or 299
        for XCeption network.
    img_ids: list of image IDs to get (optional)
        In case you only want to get certain images, supply a list of IDs to
        load. If None, load all images.
    amount: int
        Number of images that will be loaded. Cannot be more then amount of
        images in test or train folder. If None, load all images.

    Yields:
    -------
        img_array: numpy array
            Image loaded as NumPy array.
    '''
    if not img_ids:
        fnames = sorted(os.listdir(osp.join(DATA_DIR, train_or_test)))
        img_ids = [osp.splitext(osp.basename(img))[0] for img in fnames]
    if not amount:
        amount = len(img_ids)
    for img_id in img_ids[:amount]:
        yield read_img(img_id, train_or_test, size), img_id


def one_hot(labels, num_classes=None):
    '''One-hot encoding for labels.

    Arguments:
    ----------
    labels: list of labels
        List of strings or whatever.
    '''
    return to_categorical(LabelEncoder().fit_transform(labels), num_classes=num_classes)
