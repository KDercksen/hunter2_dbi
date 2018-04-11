import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocessor

from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input

from keras.applications.xception import preprocess_input

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from constants import NUM_CLASSES, SEED
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (BatchNormalization,
                          Dense,
                          Dropout,
                          GlobalAveragePooling2D,
                          Input)
from keras.models import Model
from keras.optimizers import SGD
from time import time
from tqdm import tqdm
from util import get_labels, get_images, one_hot
import keras.backend as K
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, accuracy_score
import time
from os import makedirs
from os.path import expanduser, exists, join

batch_size = 32
num_classes = NUM_CLASSES
INPUT_SIZE = 299

total_samples = 10222

print('Load labels...')
labels = get_labels()

y_train = one_hot(labels['breed'].values, num_classes=NUM_CLASSES)


##Rename name of file, these are the features from the 10222 images from the train set.
train_features = np.load('train_features_100_xception.npy')

features_train, features_validation, labels_train, labels_validation = train_test_split(
train_features[0:total_samples], y_train[0:total_samples], test_size=0.2)

print('labels shape', labels_validation.shape)

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)

logreg.fit(features_train, (labels_train * range(num_classes)).sum(axis=1))
probas = logreg.predict_proba(features_validation)

#avgprobas = np.average(probas, axis=0)

print('avgprobas shape', probas.shape)



print('ensemble validation logLoss : {}'.format(log_loss(labels_validation, probas)))
print(f'ensemble validation accuracy: {accuracy_score(np.argmax(labels_validation, axis=1), np.argmax(probas, axis=1))}')



#Rename name of file, these are the features from the 10357 images from the test set.
test_features = np.load('test_features_100_xception.npy')

predictions = logreg.predict_proba(test_features)

'''Test
This part seems a bit strange because I load the images even though I am loading the already extracted features
But this is just my lack of skills with Python.'''
nr_predictions = 10357

# Get ids, label names and images
print('Load data...')
#y_train = one_hot(labels['breed'].values, num_classes=NUM_CLASSES)
labels_predict = get_labels().sort_values(by=['breed']).breed.unique()
ids = []
images = np.zeros((nr_predictions, INPUT_SIZE, INPUT_SIZE, 3), dtype='float16')
for i, (img, img_id) in tqdm(enumerate(get_images('test', INPUT_SIZE))):
    x = preprocess_input(np.expand_dims(img, axis=0))
    images[i] = x
    ids.append(img_id)

# Save to csv
print('Saving predictions...')
with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['id'] + [l for l in labels_predict]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for i, prediction in enumerate(predictions):
        row = [ids[i]] + [f'{p:.10f}' for p in prediction]
        writer.writerow(row)

