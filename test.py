#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications import inception_v3
from keras.models import load_model
from tqdm import tqdm
from util import get_labels, get_images
import numpy as np

# Define constants
INPUT_SIZE = 299
fname = 'model1.h5'

# Load labels
print('Load labels...')
labels = get_labels()
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0) \
                                                    .drop(columns=['id'])
# Load images
print('Load images...')
images = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images('train', INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
    images[i] = x

# Load model weights
print(f'Load model from {fname}')
model = load_model(fname)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Evaluate model on data
print('Evaluate model...')
loss, accuracy = model.evaluate(images, labels_pivot.values)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')