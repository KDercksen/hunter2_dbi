#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import libraries:
from constants import SEED
from extract_features import networks
from sklearn.externals import joblib
from sklearn.metrics import log_loss, accuracy_score
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
# model = {}

for net in ['inceptionv3', 'resnet50']:
	print(f'Loading training data for {net}...')
	with open(f'bottleneck_features/{net}_avg_features_train.npy', 'rb') as f:
		x_train = np.load(f)
		print(f'Features shape: {x_train.shape}')

		le = LabelEncoder()
	le.fit(labels['breed'])
	y_train = le.transform(labels['breed'])
	
	# model[f'{net}'] = {}
	# model[f'{net}']['predict_proba'] = {}
	# model[f'{net}']['val_info'] = {}
	
	print('Creating train/val split...')
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
													test_size=.1,
													stratify=y_train)

	weight_vector = compute_class_weight('balanced', np.unique(y_train), y_train)
	class_weight = {c: w for c, w in zip(np.unique(y_train), weight_vector)}
	
	print('Training GBM and running predictions on average bottleneck features using lightgbm...')
	train_data = lgb.Dataset(x_train,label = y_train)
	test_data = lgb.Dataset(x_val,label = y_val,reference = train_data)
	
	#Setting parameters for lightgbm 
	params = {'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'multiclass',
		'num_class':120,
		'metric': 'multi_logloss',
		'learning_rate': 0.006,
		'max_depth': 9,
		'num_leaves': 3,
		'min_data_in_leaf':2,
		'feature_fraction': 0.45,
		'bagging_fraction': 0.7,
		'bagging_freq': 15,
		'max_bin':63,
		'lambda_l2': 0.1,
		'device': 'gpu'}
	
	num_round=2600
	valids = test_data
	
	start=datetime.now()
	lgbm=lgb.train(params,
		train_data,
		num_round,
		valid_sets = test_data,  # eval training data,
		verbose_eval = 10,
		early_stopping_rounds = 40
		)
	stop=datetime.now()
	
	#Execution time of the model
	execution_time_lgbm = stop-start
	execution_time_lgbm
	
	print('Done fitting')
	preds = lgbm.predict(x_val)
	# model[f'{net}']['predict_proba'] = lgbm.predict(x_val)
	# model[f'{net}']['val_info'] = y_val
	best_score = lgbm.best_score["valid_0"]["multi_logloss"]
	print(f'Best score {best_score}')
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
	store_model = f'gbm_models/{net}_gbm_{n_estimators}_acc={acc:.4f}_loss={best_score:.4f}.pkl'
	joblib.dump(lgbm, store_model)
	store_lgbm = f'gbm_models/{net}_gbm_{n_estimators}_acc={acc:.4f}_loss={best_score:.4f}.txt'
	lgbm.save_model(store_lgbm,num_iteration = lgbm.best_iteration)
	# store_model_data = f'gbm_models/{net}_model_data.pkl'
	# joblib.dump(model, store_model)
