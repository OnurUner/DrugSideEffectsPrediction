#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:04:45 2017

@author: onur
"""
import sys
import os
sys.path.append('../data/')

import numpy as np
from make_dataset import get_train_test_set, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import threading
import Queue
import cPickle as pickle
procedure_count = 200


queue = Queue.Queue()
current_path = os.path.dirname(os.path.realpath(__file__))
stability_selection_order_path = os.path.join(current_path, "./stability_selection_order.pkl")
logistic_coeff_order_path = os.path.join(current_path, "./logistic_coeff_order.pkl")
feature_importance_order_path = os.path.join(current_path, "./feature_importance_order.pkl")

def add_logistic_coefficients(x_train, y_train, selected_label):
	clf = LogisticRegression(penalty='l1', class_weight="balanced")
	clf.fit(x_train, y_train[:, selected_label])
	queue.put((selected_label, clf.coef_))
	queue.task_done()
	

def calc_feature_importance():
	task_list = []
	coefficients_per_label = dict()
	total_coeff_count = 0
	for i in xrange(procedure_count):
		print "Procedure:", i
		x_train, y_train, train_rows, x_test, y_test, test_rows, label_indexes, ADRs = get_train_test_set(validation_rate=0.25)
		for selected_label in range(y_train.shape[1]):
			if np.std(y_train[:, selected_label]) == 0:
				continue

			t = threading.Thread(target=add_logistic_coefficients,args=(x_train, y_train, selected_label))
			t.setDaemon(True)
			t.start()
			task_list.append(t)
			total_coeff_count += 1
			
	print "Finished"
	
	count = 0
	while True:
		try:
			(selected_label, coeff) = queue.get(timeout=5)
			if selected_label not in coefficients_per_label:
				coefficients_per_label[selected_label] = []
			coefficients_per_label[selected_label].append(coeff)
			print count
			count += 1
		except Exception as err:
			print "ERROR:", err
			break
	print "Total", total_coeff_count
	print "Proccessed", count
	
	for task in task_list:
		task.join()
	return coefficients_per_label

def stability_selection():
	coefficients_per_label = calc_feature_importance()
	feature_order = dict()
	for label in coefficients_per_label:
		coeff = np.vstack(coefficients_per_label[label])
		coeff_avg = np.average(coeff, axis=0)
		feature_order[label] = np.argsort(-coeff_avg)
		
		
	with open(stability_selection_order_path, 'wb') as fp:
		pickle.dump(feature_order, fp)

def logistic_coeff_selection():
	coefficients_per_label = dict()
	x, y, drug_names_data, label_indexes, ADRs = load_dataset()
	for selected_label in range(y.shape[1]):
		if np.std(y[:, selected_label]) == 0:
			continue
		clf = LogisticRegression(penalty='l1', class_weight="balanced")
		clf.fit(x, y[:,selected_label])
		coefficients_per_label[selected_label] = clf.coef_
	
	feature_order = dict()
	for label in coefficients_per_label:
		feature_order[label] = np.argsort(-coefficients_per_label[label])
		
	with open(logistic_coeff_order_path, 'wb') as fp:
		pickle.dump(feature_order, fp)
	
def feature_selection(method="stability", top=50):
	if method == "stability":
		path = stability_selection_order_path
	elif method == "logistic":
		path = logistic_coeff_order_path
	elif method == "forest":
		path = feature_importance_order_path
		
	with open(path, "rb") as input_file:
		feature_order = pickle.load(input_file)

	order = dict()
	for label in feature_order:
		order[label] = feature_order[label].ravel()[:top]
	return order

def random_forest_feature_importance():
	feature_importance_per_label = dict()
	x, y, drug_names_data, label_indexes, ADRs = load_dataset()
	for selected_label in range(y.shape[1]):
		if np.std(y[:, selected_label]) == 0:
			continue
		print "Label:", selected_label
		clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
		clf.fit(x, y[:,selected_label])
		feature_importance_per_label[selected_label] = clf.feature_importances_
	
	feature_order = dict()
	for label in feature_importance_per_label:
		feature_order[label] = np.argsort(-feature_importance_per_label[label])
		
	with open(feature_importance_order_path, 'wb') as fp:
		pickle.dump(feature_order, fp)

if __name__ == '__main__':
#	stability_selection()
#	logistic_coeff_selection()
	random_forest_feature_importance()
		
		