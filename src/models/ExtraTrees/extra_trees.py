#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
sys.path.append('../../data/')

from make_dataset import load_dataset
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

prune_count = 1

current_path = os.path.dirname(os.path.realpath(__file__))
extra_trees_result_path = current_path + '/../../../results/extra_trees_results.txt'

def save_mean_auc(auc_scores, save_path):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
		
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	f = open(save_path, "w")
	for i in sorted_means:
		f.write(str(i) + " " + str(mean_auc_scores[i]) + "\n")
	f.close()
	
def list_to_nparray(y):
	t_y = y
	for i, column in enumerate(t_y):
		t_y[i] = column[:,1]
	return np.column_stack(t_y)

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
		clf = ExtraTreesClassifier(n_estimators=100, class_weight="balanced")
		clf.fit(x_train, y_train)
		print "Calculating"
		y_probs = list_to_nparray(clf.predict_proba(x_test))
		scores = roc_auc_score(y_test, y_probs, average=None)
		
		for i, label_index in enumerate(label_indexes):
			side_effect = ADRs[label_index]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(scores[i])
			
	save_mean_auc(auc_scores, extra_trees_result_path)