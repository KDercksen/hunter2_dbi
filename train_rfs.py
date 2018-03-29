#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import SEED
from extract_features import networks
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from util import get_labels
import numpy as np

np.random.seed(seed=SEED)

n_estimators = 1000

labels = get_labels()

for net in networks.keys():
    print(f'Loading training data for {net}...')
    with open('bottleneck_features/inceptionv3_features_train.npy', 'rb') as f:
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

    # Train RF
    print('Training RF and running predictions...')
    rf = RandomForestClassifier(n_estimators=n_estimators, verbose=1, n_jobs=-1,
                                class_weight=class_weight)
    rf.fit(x_train, y_train)
    preds = rf.predict(x_val)

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
    joblib.dump(rf, store_model)
