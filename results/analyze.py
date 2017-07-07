#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from numpy import genfromtxt
import csv

logistic_regression_results_path = "logistic_regression_results.txt"
tf_logistic_regression_results_path = "tf_logistic_regression_results.txt"
keras_logistic_regression_results_path = "keras_logistic_regression_results.txt"
paper_results_path = "paper_results.txt"

if __name__ == '__main__':
	paper_adr_names = genfromtxt(paper_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	adr_names = genfromtxt(logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	tf_adr_names = genfromtxt(tf_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	keras_adr_names = genfromtxt(keras_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()

	paper_mean_auc = genfromtxt(paper_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	mean_auc = genfromtxt(logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	tf_mean_auc = genfromtxt(tf_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	keras_mean_auc = genfromtxt(keras_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	
	# TF VS SCIKITLEARN
	ofile  = open('common_scikit_tf.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Scikit-Learn Mean AUC', 'Tensorflow Mean AUC') )
	for i, adr_name in enumerate(adr_names):
		if adr_name in tf_adr_names:
			writer.writerow((adr_name, mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)]))	
	ofile.close()
	
	# KERAS VS TF
	ofile  = open('common_keras_tf.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Keras Mean AUC', 'Tensorflow Mean AUC') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in tf_adr_names:
			writer.writerow((adr_name, keras_mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)]))	
	ofile.close()
	
	# SCIKITLEARN VS PAPER
	ofile  = open('common_scikit_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Scikit-Learn Mean AUC', 'Paper Mean AUC') )
	for i, adr_name in enumerate(adr_names):
		if adr_name in paper_adr_names:
			writer.writerow((adr_name, mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)]))	
	ofile.close()
	
	# TF VS PAPER
	ofile  = open('common_tf_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Tensorflow Mean AUC', 'Paper Mean AUC') )
	for i, adr_name in enumerate(tf_adr_names):
		if adr_name in paper_adr_names:
			writer.writerow((adr_name, tf_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)]))
			
	# KERAS VS PAPER
	ofile  = open('common_keras_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Keras Mean AUC', 'Paper Mean AUC') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in paper_adr_names:
			writer.writerow((adr_name, keras_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)]))
	ofile.close()
	
	# KERAS VS TF VS SCIKITLEARN
	ofile  = open('common_keras_tf_scikit.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR_name', 'Keras_Mean_AUC', 'Tensorflow_Mean_AUC', 'Scikit-Learn_Mean_AUC') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in tf_adr_names and adr_name in adr_names:
			writer.writerow((adr_name, keras_mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)], mean_auc[adr_names.index(adr_name)]))
	ofile.close()