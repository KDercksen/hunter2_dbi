#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from constants import SEED
from extract_features import networks
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from time import time
from util import get_labels
import numpy as np

np.random.seed(seed=SEED)

n_estimators = 50
bs_size = .3
voting = 'soft'

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
                                                      test_size=.05,
                                                      stratify=y_train)

    starttime = time()
    idxs = list(range(x_train.shape[0]))
    classifiers = []
    for i in range(n_estimators):
        print(f'Training {i+1}/{n_estimators}...')
        local_starttime = time()
        # bootstrap sample indices
        sample_idxs = np.random.choice(idxs, int(x_train.shape[0]*bs_size))
        lr = LogisticRegression(solver='sag', n_jobs=-1)
        lr.fit(x_train[sample_idxs], y_train[sample_idxs])
        local_endtime = time()
        classifiers.append((f'lr {i}', lr))
        print(f'Done! Elapsed time: {local_endtime-local_starttime}')

    endtime = time()
    print(f'All classifiers fitted. Total elapsed time: {endtime-starttime}')

    voter = VotingClassifier(classifiers, voting=voting)

    acc = voter.score(x_val, y_val)
    print(f'{net} accuracy: {acc}')

    # Clear data to prevent ram issues
    x_train = None
    x_val = None
    y_train = None
    y_val = None
    preds = None

    # Store to file
    store_model = f'bag_models/{net}_voter_LR_{n_estimators}_acc={acc:.4f}.pkl'
    joblib.dump(voter, store_model)
