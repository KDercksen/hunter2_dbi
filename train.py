#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import NUM_CLASSES, SEED
from keras.applications import inception_v3
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from time import time
from tqdm import tqdm
from util import get_labels, get_images
import numpy as np

# Define constants
fname = 'model1.h5'
log_dir = f'./training_log/{time()}'
np.random.seed(seed=SEED)
INPUT_SIZE = 299
n_epochs = 100
batch_size = 32

# Load labels
print('Load labels...')
labels = get_labels()
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0) \
                                                    .drop(columns=['id'])

# Load training data
print('Load training data...')
images = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images('train', INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
    images[i] = x

# Split into training (~90%) and validation set (~10%)
print('Create train/val split...')
rnd = np.random.random(len(labels))
train_idx = rnd < 0.9
valid_idx = rnd >= 0.9

y_train = labels_pivot.values[train_idx]
y_valid = labels_pivot.values[valid_idx]

x_train = images[train_idx]
x_valid = images[valid_idx]

# Define model:
#   Add a single fully connected layer on top of the conv layers of Inception
#   Freeze Inception layers

print('Define and fit model...')
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.3)(x)
y = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=y)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit model on data, with callbacks to save best model and run TensorBoard
cp = ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)
tb = TensorBoard(log_dir=log_dir)
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), verbose=1,
          epochs=n_epochs, callbacks=[cp, tb], batch_size=batch_size)

# Evaluate predictions
print('Evaluate model...')
loss, accuracy = model.evaluate(x_valid, y_valid, batch_size=batch_size,
                                verbose=1)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')