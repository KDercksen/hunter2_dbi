#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import libraries:
from constants import SEED
from extract_features import networks
from sklearn.externals import joblib
from sklearn.metrics import log_loss, accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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

	w_vec = compute_class_weight('balanced', np.unique(y_train), y_train)
	cw = {c: w for c, w in zip(np.unique(y_train), w_vec)}
	
	print('Training GBM and running predictions on average bottleneck features using SVM...')
		
	print('Fitting svm ...')
	svm = svm.SVC(C=1.0, kernel='rbf',
		degree=3, 
		gamma='auto',
		coef0=0.0, shrinking=True, 
		probability=False,
		tol=0.001, 
		cache_size=200,
		class_weight=None,
		verbose=False, 
		max_iter=-1,
		decision_function_shape='ovr',
		random_state=None)
		
	start=datetime.now()
	svm.fit(x_train,y_train)
	stop = datetime.now()
	
	preds = svm.predict(x_val)
	
	print(preds)
	acc = accuracy_score(y_val, preds)
	
	print('Accuracy')
	print(f'{net} accuracy: {acc}')
	
	# Clear data to prevent ram issues
	x_train = None
	x_val = None
	y_train = None
	y_val = None
	preds = None

	# Store to file
	store_model = f'lda_models/{net}_rf_{n_estimators}_acc={acc:.4f}.pkl'
	joblib.dump(lda, store_model)
		
	start=datetime.now()
	lda.fit(x_train,y_train)
	stop = datetime.now()
	
	preds = lda.predict(x_val)
	prob_preds = lda.predict_proba(x_val)
	
	print(preds)
	acc = accuracy_score(y_val, preds)
	
	logloss = log_loss(y_val, prob_preds)

	print('Accuracy and log loss')
	print(f'{net} accuracy: {acc}')
	print(f'{net} log loss: {logloss}')
	
	#QDA is downright horrible...
	#Too many features, and no n_components option
	
	# Clear data to prevent ram issues
	x_train = None
	x_val = None
	y_train = None
	y_val = None
	preds = None

	# Store to file
	store_model = f'lda_models/{net}_rf_{n_estimators}_acc={acc:.4f}.pkl'
	joblib.dump(lda, store_model)