#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../../data/')
from make_dataset import load_dataset
from tflogisticnet import TFLogisticNet
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

prune_count = 1

def calc_mean_auc(auc_scores):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
	return mean_auc_scores

if __name__ == '__main__':
	X, y, sample_names, _, ADRs = load_dataset()
	kf = KFold(n_splits=3)
	
	auc_scores = dict()
	for train_index, test_index in kf.split(X, y):
		x_train = X[train_index]
		y_train = y[train_index]
		
		x_test = X[test_index]
		y_test = y[test_index]
		
		label_indexes = []
		for i in range(y.shape[1]):
		    if np.sum(y_train[:,i]) >= prune_count and np.sum(y_test[:,i]) >= prune_count:
		        label_indexes.append(i)
	
		y_train = y_train[:, label_indexes]
		y_test = y_test[:, label_indexes]
		
		net = TFLogisticNet(x_train.shape[1], y_train.shape[1])
		net.create_model()
		net.fit(x_train, y_train, epochs=50, batch_size=64, is_balanced=True)
		print "Training Finished."
		y_probs = net.predict_proba(x_test)
		print "Prediction Completed."		
		scores = roc_auc_score(y_test, y_probs, average=None)
		
		for i, label_index in enumerate(label_indexes):
			side_effect = ADRs[label_index]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(scores[i])

	mean_auc_scores = calc_mean_auc(auc_scores)
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	for i in sorted_means[:100]:
		print i, mean_auc_scores[i]