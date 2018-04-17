#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from util import get_labels
import csv
import lightgbm as lgb
import numpy as np
import os

labels = get_labels().sort_values(by=['breed']).breed.unique()
ids = [os.path.splitext(f)[0] for f in os.listdir('data/test')]

test_data = np.load('bottleneck_features/inceptionresnetv2_avg_features_test.npy')

# Load LR bag
lrbag_file = os.listdir('bag_models')[0]
print(f'LR bag: {lrbag_file}')
lrbag = joblib.load(os.path.join('bag_models', lrbag_file))

# Load GBM
gbm_file = os.listdir('gbm_models')[0]
print(f'GBM: {gbm_file}')
gbm = lgb.Booster(model_file=os.path.join('gbm_models', gbm_file))

# Run predictions
lrbag_preds = lrbag.predict_proba(test_data)
print(lrbag_preds.shape)
np.save('predictions/raw_lrbag.npy', lrbag_preds)

gbm_preds = gbm.predict(test_data)
print(gbm_preds.shape)
np.save('predictions/raw_gbm.npy', gbm_preds)

# Combine predictions
comb_preds = np.zeros(lrbag_preds.shape)
for i in range(lrbag_preds.shape[0]):
    lr_p = lrbag_preds[i].max()
    gbm_p = gbm_preds[i].max()
    if lr_p > gbm_p:
        comb_preds[i] = lrbag_preds[i]
    else:
        comb_preds[i] = gbm_preds[i]

# Output to csv
np.save('predictions/combined.npy', comb_preds)
with open('predictions.csv', 'w') as f:
    fieldnames = ['id'] + [l for l in labels]
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for i, prediction in enumerate(comb_preds):
        row = [ids[i]] + [f'{p:.10f}' for p in prediction]
        writer.writerow(row)
