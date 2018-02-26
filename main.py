#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import NUM_CLASSES, SEED
from keras.applications import inception_v3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.metrics import log_loss
from tqdm import tqdm
from util import get_labels, get_images
import numpy as np

'''Define constants'''
np.random.seed(seed=SEED)
INPUT_SIZE = 299

'''Load labels'''
print('Load labels...')
labels = get_labels()
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0) \
                                                    .drop(columns=['id'])

'''Load training data'''
print('Load training data...')
images = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, img in tqdm(enumerate(get_images('train', INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
    images[i] = x

'''Split into training (~80%) and validation set (~20%)'''
print('Create train/val split...')
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8

y_train = labels_pivot.values[train_idx]
y_valid = labels_pivot.values[valid_idx]

x_train = images[train_idx]
x_valid = images[valid_idx]

'''Define model:
    Add a single fully connected layer on top of the conv layers of Inception
    Freeze Inception layers
'''
print('Define and fit model...')
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
y = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=y)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

'''Fit model on data'''
model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

'''Save model'''
fname = 'model1.h5'
print(f'Saving model to {fname}')
model.save(fname)

'''Predict validation set'''
print('Predict val set...')
pred_valid = model.predict(x_valid, batch_size=32, verbose=1)

'''Evaluate predictions'''
print(f'Validation log loss {log_loss(y_valid, pred_valid)}')

# TODO: accuracy on validation set