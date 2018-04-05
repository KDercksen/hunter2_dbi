#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import libraries:
from constants import SEED
from extract_features import networks
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from util import get_labels

from xgboost import XGBClassifier
import matplotlib.pylab as plt
#matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import numpy as np
import pandas as pd

np.random.seed(seed=SEED)

n_estimators = 100

labels = get_labels()


for net in networks.keys():
    print(f'Loading training data for {net}...')
    with open('bottleneck_features_avg/densenet121_avg_features_train.npy', 'rb') as f:
        x_train = np.load(f)
        print(f'Features shape: {x_train.shape}')

    le = LabelEncoder()
    le.fit(labels['breed'])
    y_train = le.transform(labels['breed'])

    print('Creating train/val split...')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=.1,
                                                      stratify=y_train)

    weight_vector = compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight = {c: w for c, w in zip(np.unique(y_train), weight_vector)}

    # Train GBM
    # print('Training GBM and running predictions on average bottleneck features...')
    # gbm = GradientBoostingClassifier(n_estimators=n_estimators, 
													# max_features = 50,
													# max_depth = 2,
													# subsample = 0.6,
													# verbose=2)
    # gbm.fit(x_train, y_train)
	# preds = gbm.predict(x_val)
    
    # NEEDS TUNING/EDITING, SEEMS VERY SLOW EVEN WITH POOLED FEATURES!!!
    clf = XGBClassifier(
    learning_rate =0.1,
    n_estimators=20,
    max_depth=1,
    min_child_weight=1,
    gamma=0,
    subsample=0.3,
    colsample_bytree=0.5,
    objective= 'multi:softmax',
    nthread = 7,
    scale_pos_weight=1,
    seed=27,
    silent = True)


    print('Training GBM and running predictions on average bottleneck features using xgboost...')
    clf.fit(x_train, y_train,verbose = True)
    print('Done fitting')
    preds = clf.predict(x_val)

    acc = accuracy_score(y_val, preds)
    print(f'{net} accuracy: {acc}')

    # Clear data to prevent ram issues
    x_train = None
    x_val = None
    y_train = None
    y_val = None
    preds = None

    # Store to file
    store_model = f'rf_models/{net}_rf_{n_estimators}_acc={acc:.4f}.pkl'
    joblib.dump(gbm, store_model)
