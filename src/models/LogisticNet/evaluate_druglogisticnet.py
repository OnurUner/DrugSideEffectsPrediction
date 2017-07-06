#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from sklearn.metrics import hamming_loss, accuracy_score
import scipy.io
import numpy as np
import train_druglogisticnet
from ml_metrics import mapk

results_path = train_druglogisticnet.results_path

def evaluate(y_true, y_pred, y_pred_prob):	
	print "Jaccard similarity coefficient score:", jaccard_similarity_score(y_true, y_pred)
	print "Hamming Loss:", hamming_loss(y_true, y_pred)
	print "F1 Score:", f1_score(y_true, y_pred, average="samples")
	print "Log loss:", log_loss(y_true, y_pred_prob)
	print "Average Precision Score(samples):", average_precision_score(y_true, y_pred_prob, average='samples')
	
	truths = get_indexes(y_true)
	predictions = get_indexes(y_pred)
	print "Mean average precision score:", mapk(truths, predictions, k=10)
	

def get_indexes(y):
	indexes = []
	for r in y:
		indexes.append(np.where(r == 1)[0].tolist())
	return indexes

if __name__ == '__main__':
	results = scipy.io.loadmat(results_path)
	y_pred = results["y_pred"]
	y_true = results["y_true"]
	y_pred_prob = results["y_pred_prob"]

	ham_loss = hamming_loss(y_true, y_pred)
	
	accuracies = list()
	for i in range(y_pred.shape[1]):
		acc = accuracy_score(y_true[:,i], y_pred[:,i])
		accuracies.append(acc)
	
	print "Average of accuracies:", np.mean(accuracies)
	evaluate(y_true, y_pred, y_pred_prob)