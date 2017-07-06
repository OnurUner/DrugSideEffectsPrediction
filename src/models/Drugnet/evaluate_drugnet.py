#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, log_loss, average_precision_score, jaccard_similarity_score, classification_report
import train_drugnet
from ml_metrics import mapk

results_path = train_drugnet.results_path

def run():
	mat = scipy.io.loadmat(results_path)
	scores = mat["scores"].ravel()

	y_pred_prob = mat["y_pred_prob"]
	y_true = mat["y_true"]
	
	y_pred_prob = y_pred_prob.reshape((3165,237)).T
	y_pred = np.array(y_pred_prob)
	y_true = np.array(y_true).reshape((3165,237)).T
	y_pred[y_pred >= 0.5] = 1
	y_pred[y_pred <= 0.5] = 0

	acc = scores[3166:]
	print "Average of accuracies: ", np.mean(acc)
	print "Drugnet results:"
#	evaluate(y_true, y_pred, y_pred_prob)
	
	print "\n-----------------------------\n"
	y_pred.fill(0)
	y_pred_prob.fill(0)
	acc = []
	for i in range(3165):
		acc.append(accuracy_score(y_true[:,i], y_pred[:,i]))
	print np.mean(acc)

	print "Zero prediction results:"
	evaluate(y_true, y_pred, y_pred_prob)

def get_indexes(y):
	indexes = []
	for r in y:
		indexes.append(np.where(r == 1)[0].tolist())
	return indexes


def evaluate(y_true, y_pred, y_pred_prob):	
	print "Jaccard similarity coefficient score:", jaccard_similarity_score(y_true, y_pred)
	print "Hamming Loss:", hamming_loss(y_true, y_pred)
	print "F1 Score:", f1_score(y_true, y_pred, average="samples")
	print "Log loss:", log_loss(y_true, y_pred_prob)
	print "Average Precision Score(samples):", average_precision_score(y_true, y_pred_prob, average='samples')

	
	truths = get_indexes(y_true)
	predictions = get_indexes(y_pred)
	print "Mean average precision score:", mapk(truths, predictions, k=10)

if __name__ == '__main__':
	run()