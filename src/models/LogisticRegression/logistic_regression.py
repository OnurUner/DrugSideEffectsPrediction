#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
sys.path.append('../../data/')
sys.path.append('../../utils/')

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from generate_folds import load_folds
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from utility import save_mean_auc, get_train_index, feature_selection

prune_count = 1

current_path = os.path.dirname(os.path.realpath(__file__))
multilabel_result_path = current_path + '/../../../results/results_filtered/is_ovr_logistic_regression_results.txt'
binary_result_path = current_path + '/../../../results/results_filtered/binary_logistic_regression_results.txt'

folds_path = current_path + '/../../log/folds/pos_sample_counts.txt'


def multilabel_classifier(X, y, sample_names, ADRs, folds):
	auc_scores = dict()
	for i, test_index in enumerate(folds):
		train_index = get_train_index(folds, i)
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
		clf = OneVsRestClassifier(LogisticRegression(C=1e5, class_weight="balanced"), n_jobs=32)
		clf.fit(x_train, y_train)
		print "Calculating"
		y_probs = clf.predict_proba(x_test)
		scores = roc_auc_score(y_test, y_probs, average=None)
		
		for i, label_index in enumerate(label_indexes):
			side_effect = ADRs[label_index]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(scores[i])
			
	save_mean_auc(auc_scores, multilabel_result_path)
		

def binary_classifier(X, y, sample_names, ADRs, split_count):	
	skf = StratifiedKFold(n_splits=split_count)
	auc_scores = dict()
	foldfile = open(folds_path, "w")
	foldfile.write("Train \t Test \n")
	
	for selected_label in range(y.shape[1]):
		for train_index, test_index in skf.split(X, y[:,selected_label]):
			x_train = X[train_index]
			y_train = y[train_index, selected_label]
			
			x_test = X[test_index]
			y_test = y[test_index, selected_label]
			
			foldfile.write(str(np.sum(y_train)) + " \t " + str(np.sum(y_test)) + "\n")
			
			clf = LogisticRegression(C=1e5, class_weight="balanced")
			clf.fit(x_train, y_train)
			
			y_probs = clf.predict_proba(x_test)[:,1]
			score = roc_auc_score(y_test, y_probs, average=None)
			
			side_effect = ADRs[selected_label]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(score)
		foldfile.write( "--------------")

	save_mean_auc(auc_scores, binary_result_path)	
	foldfile.close()

if __name__ == '__main__':
	n_split = 3
	X, y, sample_names, ADRs, SOIS, IS = load_folds()
	binary_classifier(X, y, sample_names, ADRs, n_split)
	multilabel_classifier(X, y, sample_names, ADRs, IS)