#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append('../../data/')
import scipy.io
import numpy as np
from make_dataset import load_dataset
from datetime import datetime
from drugnet import DrugNet

results_path = "./results/evaluation_30epoch.mat"

def run():
	x_train, y_train, train_rows, x_test, y_test, test_rows = load_dataset(prune_count=50)
	task_count = y_train.shape[1]
	y_train = np.split(y_train, task_count, axis=1)
	y_test = np.split(y_test, task_count, axis=1)
	
	drugnet = DrugNet((x_train.shape[1],), task_count)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	
	print "Model creating..."
	drugnet.create_model()
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	
	print "Model start to train..."
	drugnet.fit(x_train, y_train)
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

	print "Model saving..."
	drugnet.save_model()
	
	print "Model evaluation..."
	metrics_names, scores = drugnet.evaluate(x_test, y_test)
	y_pred_prob = drugnet.predict(x_test)
	mat_dict = dict()
	mat_dict["scores"] = scores
	mat_dict["metrics_names"] = metrics_names
	mat_dict["y_pred_prob"] = y_pred_prob
	mat_dict["y_true"] = y_test
	scipy.io.savemat(results_path, mdict=mat_dict)
	print "Finished"
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	
	
if __name__ == '__main__':
	run()