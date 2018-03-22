#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import NUM_CLASSES, SEED
from keras.applications import (inception_v3,
                                resnet50)
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (BatchNormalization,
                          Dense,
                          Dropout,
                          GlobalAveragePooling2D,
                          Input,
                          Concatenate)
from keras.models import Model
from keras.optimizers import SGD
from time import time
from tqdm import tqdm
from util import get_labels, get_images, one_hot
import keras.backend as K
import numpy as np

# Define constants
fname = 'model1_finetune.h5'
log_dir = f'./training_log/{time()}'
np.random.seed(seed=SEED)
INPUT_SIZE = 299
n_pre_epochs = 10
n_epochs = 100
batch_size = 32
n_images = 100 #len(labels)

# Load labels
print('Load labels...')
labels = get_labels()[:n_images]

# Load training data
print('Load training data...')
x_train = np.zeros((n_images, INPUT_SIZE, INPUT_SIZE, 3), dtype=K.floatx())
for i, (img, img_id) in tqdm(enumerate(get_images('train', INPUT_SIZE, amount=n_images))):
    x = inception_v3.preprocess_input(np.expand_dims(img, axis=0))
    x_train[i] = x
y_train = one_hot(labels['breed'].values, num_classes=NUM_CLASSES)

# Arguments of ImageDataGenerator define types of augmentation to be performed
# E.g: Horizontal flip, rotation, etc...
# no fitting required since we don't use centering/normalization/whitening
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True,
    validation_split=.1)

# Define model:
#   Add a single fully connected layer on top of the conv layers of Inception
#   Freeze Inception layers
print('Define and fit model...')
#input = Input(shape=(INPUT_SIZE,INPUT_SIZE,3))
#resnet50 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE,INPUT_SIZE,3), input_tensor=input)
#r = resnet50.output
inception_v3 = inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = inception_v3.output
#print(K.shape(x))
#print(K.shape(r))
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
#r = BatchNormalization()(r)
#r = GlobalAveragePooling2D()(r)
#x = Concatenate(axis=0)([x,r])
#print(K.shape(x))
x = Dense(1024, activation='relu')(x)
x = Dropout(.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=inception_v3.input, outputs=predictions)

for layer in inception_v3.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit model on data, with callbacks to save best model and run TensorBoard
cp = ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)
tb = TensorBoard(log_dir=log_dir)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size,
                                 subset='training'),
                    validation_data=datagen.flow(x_train, y_train,
                                                 batch_size=batch_size,
                                                 subset='validation'),
                    steps_per_epoch=x_train.shape[0] / batch_size,
                    epochs=n_pre_epochs,
                    callbacks=[cp, tb])

# Now we will fine-tune the top inception block
print('Fine-tuning model')
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size,
                                 subset='training'),
                    validation_data=datagen.flow(x_train, y_train,
                                                 batch_size=batch_size,
                                                 subset='validation'),
                    steps_per_epoch=x_train.shape[0] / batch_size,
                    epochs=n_pre_epochs + n_epochs,
                    initial_epoch=n_pre_epochs,
                    callbacks=[cp, tb])
