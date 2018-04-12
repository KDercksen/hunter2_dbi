#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import libraries:
from constants import SEED
from extract_features import networks
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from util import get_labels
import lightgbm as lgb
from datetime import datetime

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
	with open(f'bottleneck_features_avg/{net}_avg_features_train.npy', 'rb') as f:
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

	print('Training GBM and running predictions on average bottleneck features using lightgbm...')
	train_data = lgb.Dataset(x_train,label = y_train)
	
	#Setting parameters for lightgbm 
	params = {'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'multiclass',
		'num_class':120,
		'metric': 'multi_logloss',
		'learning_rate': 0.002,
		'max_depth': 10,
		'num_leaves': 30,
		'min_data_in_leaf':2,
		'feature_fraction': 0.4,
		'bagging_fraction': 0.6,
		'bagging_freq': 17,
		'max_bin':63,
		'device': 'gpu'}
	
	num_round=100
	start=datetime.now()
	lgbm=lgb.train(params,train_data,num_round)
	stop=datetime.now()
	
	#Execution time of the model
	execution_time_lgbm = stop-start
	execution_time_lgbm
	
	print('Done fitting')
	preds = lgbm.predict(x_val)
	
	predictions = []

	for x in preds:
		predictions.append(np.argmax(x))
	
	acc = accuracy_score(y_val, predictions)
	print(f'{net} accuracy: {acc}')
	
	# Clear data to prevent ram issues
	x_train = None
	x_val = None
	y_train = None
	y_val = None
	preds = None

	# Store to file
	store_model = f'rf_models/{net}_rf_{n_estimators}_acc={acc:.4f}.pkl'
	joblib.dump(lgbm, store_model)