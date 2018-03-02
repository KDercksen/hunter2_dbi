#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications import inception_v3
from keras.models import load_model
from tqdm import tqdm
from util import get_labels, get_images
import csv
import numpy as np

# Define constants
INPUT_SIZE = 299
fname = 'model1.h5'
train_test = 'test'
nr_predictions = 10357

# Get ids, label names and images
print('Load data...')
labels = get_labels().sort_values(by=['breed']).breed.unique()
ids = []
images = np.zeros((nr_predictions, INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images(train_test, INPUT_SIZE))):
    x = inception_v3.preprocess_input(np.expand_dims(img.copy(), axis=0))
    images[i] = x
    ids.append(img_id)

# Load model weights
print(f'Load model from {fname}')
model = load_model(fname)

# Make predictions on input images
print('Predict...')
predictions = model.predict(images, verbose=1)

# Save to csv
print('Saving predictions...')
with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['id'] + [l for l in labels]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for i, prediction in enumerate(predictions):
        row = [ids[i]] + [f'{p:.10f}' for p in prediction]
        writer.writerow(row)