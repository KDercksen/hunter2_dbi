#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications import (inception_v3,
                                resnet50,
                                densenet,
                                xception,
                                vgg16,
                                vgg19,
                                inception_resnet_v2)
from tqdm import tqdm
from util import get_labels, get_images
import argparse
import keras.backend as K
import numpy as np
import os
import sys

# Maps network names to module, constructor, input size
networks = {
    'xception': (xception, xception.Xception, 299),
    'vgg16': (vgg16, vgg16.VGG16, 224),
    'vgg19': (vgg19, vgg19.VGG19, 224),
    'resnet50': (resnet50, resnet50.ResNet50, 224),
    'inceptionv3': (inception_v3, inception_v3.InceptionV3, 299),
    'inceptionresnetv2': (inception_resnet_v2, inception_resnet_v2.InceptionResNetV2, 299),
    'densenet121': (densenet, densenet.DenseNet121, 224),
    'densenet169': (densenet, densenet.DenseNet169, 224),
    'densenet201': (densenet, densenet.DenseNet201, 224),
}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('network', help='Network to use for feature extraction',
                   choices=networks.keys())
    p.add_argument('savedir', help='Directory to save output')
    args = p.parse_args()

    print('Running feature extraction...')
    print(f'[CNN]    {args.network}')
    print(f'[OUTPUT] {args.savedir}')

    preprocess_func = networks[args.network][0].preprocess_input
    NetworkConstr = networks[args.network][1]
    input_size = networks[args.network][2]

    # Load labels
    labels = get_labels()
    data_size = {'train': len(labels), 'test': 10357}

    model = NetworkConstr(weights='imagenet', pooling='avg', include_top=False)

    # Load process train/test data
    for key, size in data_size.items():
        save_fname = f'{args.network}_avg_features_{key}.npy'
        save_path = os.path.join(args.savedir, save_fname)
        if os.path.exists(save_path):
            print(f'{save_path} already exists, skipping!')
            continue

        print(f'Load {key} data...')
        images = np.zeros((size, input_size, input_size, 3), dtype=K.floatx())
        for i, (img, img_id) in tqdm(enumerate(get_images(key, input_size))):
            x = preprocess_func(np.expand_dims(img, axis=0))
            images[i] = x

        # Run predictions for training data and reshape to get feature vectors
        features = model.predict(images, batch_size=32, verbose=1)
        images = None
        features = features.reshape((features.shape[0], np.prod(features.shape[1:])))

        print(f'Saving to {save_path}')
        with open(save_path, 'wb') as f:
            np.save(f, features)


    print('Done!')
    sys.exit(0)
