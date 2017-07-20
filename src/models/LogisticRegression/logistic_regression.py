#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
sys.path.append('../../data/')

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from make_dataset import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

prune_count = 1

current_path = os.path.dirname(os.path.realpath(__file__))
multilabel_result_path = current_path + '/../../../results/ovr_logistic_regression_results.txt'
binary_result_path = current_path + '/../../../results/binary_logistic_regression_results.txt'

folds_path = current_path + '/../../log/folds/pos_sample_counts.txt'

def save_mean_auc(auc_scores, save_path):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
		
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	f = open(save_path, "w")
	for i in sorted_means:
		f.write(str(i) + " " + str(mean_auc_scores[i]) + "\n")
	f.close()


def multilabel_classifier(X, y, sample_names, ADRs, split_count):
	kf = KFold(n_splits=split_count)
	
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
	X, y, sample_names, _, ADRs = load_dataset(prune_count=11)
	binary_classifier(X, y, sample_names, ADRs, n_split)
	multilabel_classifier(X, y, sample_names, ADRs, n_split)