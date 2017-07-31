#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
sys.path.append('../../data/')
sys.path.append('../../utils/')

import scipy.io
from make_dataset import load_dataset
from datetime import datetime
from druglogisticnet import DrugLogisticNet
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from utility import save_mean_auc

prune_count = 1
current_path = os.path.dirname(os.path.realpath(__file__))
result_path = current_path + '/../../../results/keras_logistic_regression_results.txt'


def probs_to_dict(y_pred, outputs):
	prob_classes = dict()
	for i, output in enumerate(outputs):
		out_name = output.name
		prob_classes[out_name] = y_pred[i]
	return prob_classes

if __name__ == '__main__':
	X, y, sample_names, _, ADRs = load_dataset(prune_count=50)
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
		
		output_names = []
		for i in label_indexes:
			output_names.append(str(i))
			
		y_train_dict = dict()
		y_test_dict = dict()
		for i, out_name in enumerate(output_names):
			y_train_dict[out_name] = y_train[:,i]
			y_test_dict[out_name] = y_test[:,i]
		
		net = DrugLogisticNet(input_shape=(x_train.shape[1],), output_shape=y_train.shape[1], output_names=output_names)
		net.create_model()
		print "Model Created."
		net.fit(x_train, y_train_dict, epochs=50, batch_size=64, is_balanced=True, verbose=0)
		print "Training Finished."
		y_probs = net.predict(x_test, batch_size=x_test.shape[0])
		y_probs_dict = probs_to_dict(y_probs, net.model.output_layers)
		
		for out_name in y_probs_dict:
			t_y = y_test_dict[out_name]
			p_y = y_probs_dict[out_name]
			score = roc_auc_score(t_y, p_y, average=None)
			side_effect = ADRs[int(out_name)]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(score)
		
	save_mean_auc(auc_scores, result_path)
