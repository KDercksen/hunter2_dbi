#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
from extract_features import networks
from sklearn.externals import joblib
from util import get_labels
import csv
import lightgbm as lgb
import numpy as np
import os

labels = get_labels().sort_values(by=['breed']).breed.unique()
ids = [os.path.splitext(f)[0] for f in sorted(os.listdir('data/test'))]

# Array to store predictions
comb_preds = np.zeros((len(ids), 120))

# Load LR bags
lrbags = []
for fname in os.listdir('bag_models'):
    lrbag_file = os.path.join('bag_models', fname)
    print(f'LR bag: {lrbag_file}')
    lrbags.append((lrbag_file, joblib.load(lrbag_file)))

# Load GBMs
gbms = []
for fname in os.listdir('gbm_models'):
    gbm_file = os.path.join('gbm_models', fname)
    print(f'GBM: {gbm_file}')
    gbms.append((gbm_file, lgb.Booster(model_file=gbm_file)))

# Run predictions, keeping best one
counts = {i: '' for i in range(len(ids))}
# for net in networks.keys():
for net in ['inceptionresnetv2']:
    # Load test data for this network
    test_data = np.load(f'bottleneck_features/{net}_avg_features_test.npy')
    # Load corresponding LR bag model (TODO: extend with GBM/RF?)
    lrbag = [l for l in lrbags if net in l[0]][0][1]
    # Predictions for both models
    lrbag_preds = lrbag.predict_proba(test_data)
    # If LR bag prediction is more sure than previous prediction in array,
    # update. Else keep the old prediction.
    for i in range(lrbag_preds.shape[0]):
        p_lr = lrbag_preds[i]
        p_ex = comb_preds[i]
        if p_lr.max() > p_ex.max():
            comb_preds[i] = p_lr
            counts[i] = net

print(Counter(counts.values()))

# Output to csv
np.save('predictions/combined.npy', comb_preds)
with open('predictions.csv', 'w') as f:
    fieldnames = ['id'] + [l for l in labels]
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for i, prediction in enumerate(comb_preds):
        row = [ids[i]] + [f'{p:.10f}' for p in prediction]
        writer.writerow(row)
