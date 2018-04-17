#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import SEED
from extract_features import networks
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from time import time
from util import get_labels
import numpy as np

np.random.seed(seed=SEED)

n_estimators = 20
bs_size = .8
f_size = 1.

labels = get_labels()

for net in networks.keys():
    print(f'Loading training data for {net}...')
    with open(f'bottleneck_features/{net}_avg_features_train.npy', 'rb') as f:
        x_train = np.load(f)
        print(f'Features shape: {x_train.shape}')

    le = LabelEncoder()
    le.fit(labels['breed'])
    y_train = le.transform(labels['breed'])

    w_vec = compute_class_weight('balanced', np.unique(y_train), y_train)
    cw = {c: w for c, w in zip(np.unique(y_train), w_vec)}

    print('Creating train/val split...')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=.1,
                                                      stratify=y_train)

    print('Training bag of regressors...')
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                            class_weight=cw)
    bag = BaggingClassifier(lr, n_estimators=n_estimators, max_samples=bs_size,
                            max_features=f_size, n_jobs=-1, verbose=1)

    starttime = time()
    bag.fit(x_train, y_train)
    endtime = time()
    print(f'Done training. Elapsed time: {endtime-starttime:.2f}s')

    acc = bag.score(x_val, y_val)
    print(f'{net}: total mean accuracy: {acc}')

    # Clear data to prevent ram issues
    x_train = None
    x_val = None
    y_train = None
    y_val = None
    preds = None

    # Store to file
    store_model = f'bag_models/{net}_LRbag_{n_estimators}_acc={acc:.4f}.pkl'
    joblib.dump(bag, store_model)
