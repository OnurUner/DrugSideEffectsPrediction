#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
sys.path.append('../../data/')
sys.path.append('../../utils/')

from generate_folds import load_folds
from tflogisticnet import TFLogisticNet
from sklearn.metrics import roc_auc_score
from utility import save_mean_auc, get_train_index, feature_selection

prune_count = 1
current_path = os.path.dirname(os.path.realpath(__file__))
result_path = current_path + '/../../../results/results_filtered/tf_logistic_regression_300feature_selected_results.txt'


if __name__ == '__main__':
	X, y, sample_names, ADRs, SOIS, IS = load_folds()
#	_, _, X = feature_selection(X, y, 10)
#	folds = SOIS
#	auc_scores = dict()
#	for i, test_index in enumerate(folds):
#		train_index = get_train_index(folds, i)
#		x_train = X[train_index]
#		y_train = y[train_index]
#		
#		x_test = X[test_index]
#		y_test = y[test_index]
#		
#		label_indexes = []
#		for i in range(y.shape[1]):
#		    if np.sum(y_train[:,i]) >= prune_count and np.sum(y_test[:,i]) >= prune_count:
#		        label_indexes.append(i)
#	
#		y_train = y_train[:, label_indexes]
#		y_test = y_test[:, label_indexes]
#		
#		net = TFLogisticNet(x_train.shape[1], y_train.shape[1])
#		net.create_model()
#		net.fit(x_train, y_train, epochs=50, batch_size=64, is_balanced=True)
#		print "Training Finished."
#		y_probs = net.predict_proba(x_test)
#		print "Prediction Completed."		
#		scores = roc_auc_score(y_test, y_probs, average=None)
#		
#		for i, label_index in enumerate(label_indexes):
#			side_effect = ADRs[label_index]
#			if side_effect not in auc_scores:	
#				auc_scores[side_effect] = []
#			auc_scores[side_effect].append(scores[i])
#
#	save_mean_auc(auc_scores, result_path)
