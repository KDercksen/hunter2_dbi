#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import NUM_CLASSES, SEED
from keras.applications import inception_v3
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (BatchNormalization,
                          Dense,
                          Dropout,
                          GlobalAveragePooling2D)
from keras.models import Model
from sklearn.model_selection import train_test_split
from time import time
from tqdm import tqdm
from util import get_labels, get_images, one_hot
import numpy as np

# Define constants
fname = 'model1.h5'
log_dir = f'./training_log/{time()}'
np.random.seed(seed=SEED)
INPUT_SIZE = 299
n_epochs = 200
batch_size = 32

# Load labels
print('Load labels...')
labels = get_labels()

# Load training data
print('Load training data...')
images = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images('train', INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img, axis=0))
    images[i] = x

# Split into training (~90%) and validation set (~10%)
print('Create train/val split...')
y_train = one_hot(labels['breed'].values)
x_train, x_valid, y_train, y_valid = train_test_split(images, y_train,
                                                      test_size=.1,
                                                      stratify=y_train)

# Define model:
#   Add a single fully connected layer on top of the conv layers of Inception
#   Freeze Inception layers
print('Define and fit model...')
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base_model.input, predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit model on data, with callbacks to save best model and run TensorBoard
cp = ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)
tb = TensorBoard(log_dir=log_dir)
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1,
          epochs=n_epochs, callbacks=[cp, tb], batch_size=batch_size)
