#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications import inception_v3
from keras.models import load_model
from tqdm import tqdm
from util import get_labels, get_images, one_hot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss

# Define constants
INPUT_SIZE = 299
fname = 'model1.h5'
fconf = 'conf_mat1.csv'

# Load labels
print('Load labels...')
labels = get_labels()

# Load images
print('Load images...')
images = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images('train', INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img, axis=0))
    images[i] = x

# Load one-hot encodings
y_train = one_hot(labels['breed'].values)

# Load model weights
print(f'Load model from {fname}')
model = load_model(fname)

# Evaluate model on data
print('Predict...')
predictions = model.predict(images, verbose=1)

print(f'Loss: {log_loss(y_train, predictions)}')
print(f'Accuracy: {accuracy_score(y_train, np.argmax(predictions, axis=1))}')
conf_arr = confusion_matrix(y_train, np.argmax(predictions, axis=1))

print(f'Save confusion matrix to {fconf}')
np.savetxt(fconf, conf_arr, delimiter=",")