#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../../data/')
import scipy.io
from make_dataset import load_dataset
from datetime import datetime
from druglogisticnet import DrugLogisticNet
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

prune_count = 1
#results_path = "./results/evaluation_30epoch.mat"

#def run():
#	x_train, y_train, train_rows, x_test, y_test, test_rows, label_indexes = get_train_test_set(validation_rate=0.33, prune_count=1)
#	input_shape = (x_train.shape[1],)
#	output_shape = y_train.shape[1]
#	output_names = []
#	for i in label_indexes:
#		output_names.append('class_'+str(i))
#	 
#	drugnet = DrugLogisticNet(input_shape=input_shape, output_shape=output_shape, output_names=output_names)
#	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#	
#	print "Model creating..."
#	drugnet.create_model()
#	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#	
#	print "Model start to train..."
#	drugnet.fit(x_train, y_train, epochs=100, is_balanced=True)
#	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#
#	print "Model saving..."
#	drugnet.save_model()
#	
#	print "Model evaluation..."
#	metrics_names, scores = drugnet.evaluate(x_test, y_test, batch_size=237)
#	y_pred_prob = drugnet.predict(x_test)
#
#	y_pred = np.array(y_pred_prob)
#	y_pred[y_pred>=0.5] = 1
#	y_pred[y_pred<0.5] = 0
#
#	mat_dict = dict()
#	mat_dict["scores"] = scores
#	mat_dict["metrics_names"] = metrics_names
#	mat_dict["y_pred_prob"] = y_pred_prob
#	mat_dict["y_pred"] = y_pred
#	mat_dict["y_true"] = y_test
#	scipy.io.savemat(results_path, mdict=mat_dict)
#	print "Finished"
#	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def calc_mean_auc(auc_scores):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
	return mean_auc_scores

def probs_to_dict(y_pred, outputs):
	prob_classes = dict()
	for i, output in enumerate(outputs):
		out_name = output.name
		prob_classes[out_name] = y_pred[i]
	return prob_classes

if __name__ == '__main__':
	X, y, sample_names, _ = load_dataset()
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
			output_names.append('class_'+str(i))
			
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
			if out_name not in auc_scores:	
				auc_scores[out_name] = []
			auc_scores[out_name].append(score)
		
		del net.model
		del net
		
	mean_auc_scores = calc_mean_auc(auc_scores)
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	for i in sorted_means[:5]:
		print i, mean_auc_scores[i]