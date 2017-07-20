#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from numpy import genfromtxt
import csv

where = "results_filtered"

ovr_logistic_regression_results_path = "./"+where+"/ovr_logistic_regression_results.txt"
binary_logistic_regression_results_path = "./"+where+"/binary_logistic_regression_results.txt"
tf_ovr_logistic_regression_results_path = "./"+where+"/tf_logistic_regression_results.txt"
keras_ovr_logistic_regression_results_path = "./"+where+"/keras_logistic_regression_results.txt"
extra_tress_results_path = "./"+where+"/extra_trees_results.txt"
paper_results_path = "./"+where+"/paper_results.txt"
web_all_results_path = "./"+where+"/web_auc_all.txt"
web_filtered_results_path = "./"+where+"/web_auc.txt"

pos_count_path = "./"+where+"/pos_counts.txt"

if __name__ == '__main__':
	paper_adr_names = genfromtxt(paper_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	ovr_adr_names = genfromtxt(ovr_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	tf_adr_names = genfromtxt(tf_ovr_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	keras_adr_names = genfromtxt(keras_ovr_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	binary_adr_names = genfromtxt(binary_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	extra_trees_adr_names = genfromtxt(extra_tress_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	pos_counts_adr_names = genfromtxt(pos_count_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	web_all_adr_names = genfromtxt(web_all_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	web_filtered_adr_names = genfromtxt(web_all_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	
	paper_mean_auc = genfromtxt(paper_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	ovr_mean_auc = genfromtxt(ovr_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	tf_mean_auc = genfromtxt(tf_ovr_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	keras_mean_auc = genfromtxt(keras_ovr_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	binary_mean_auc = genfromtxt(binary_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	extra_tress_mean_auc = genfromtxt(extra_tress_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	pos_counts = genfromtxt(pos_count_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	web_all_mean_auc = genfromtxt(web_all_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	web_filtered_mean_auc = genfromtxt(web_filtered_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	
	
	# TF VS SCIKITLEARN
	ofile  = open('common_scikit_tf.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Scikit-Learn Mean AUC', 'Tensorflow Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(ovr_adr_names):
		if adr_name in tf_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, ovr_mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)], pos_counts[pos_count_index]))	
	ofile.close()
	
	# KERAS VS TF
	ofile  = open('common_keras_tf.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Keras Mean AUC', 'Tensorflow Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in tf_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, keras_mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)], pos_counts[pos_count_index]))	
	ofile.close()
	
	# SCIKITLEARN VS PAPER
	ofile  = open('common_scikit_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Scikit-Learn Mean AUC', 'Paper Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(ovr_adr_names):
		if adr_name in paper_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, ovr_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)], pos_counts[pos_count_index]))	
	ofile.close()
	
	# TF VS PAPER
	ofile  = open('common_tf_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Tensorflow Mean AUC', 'Paper Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(tf_adr_names):
		if adr_name in paper_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, tf_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)], pos_counts[pos_count_index]))
			
	# KERAS VS PAPER
	ofile  = open('common_keras_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Keras Mean AUC', 'Paper Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in paper_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, keras_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)], pos_counts[pos_count_index]))
	ofile.close()
	
	# KERAS VS TF VS SCIKITLEARN
	ofile  = open('common_keras_tf_scikit.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Keras Mean AUC', 'Tensorflow Mean AUC', 'Scikit-Learn Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(keras_adr_names):
		if adr_name in tf_adr_names and adr_name in ovr_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, keras_mean_auc[i], tf_mean_auc[tf_adr_names.index(adr_name)], ovr_mean_auc[ovr_adr_names.index(adr_name)], pos_counts[pos_count_index]))
	ofile.close()
	
	# BINARY VS MULTIPLE
	ofile  = open('common_binary_multiple.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Binary Clf Mean AUC', 'Multilabel Clf Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(binary_adr_names):
		if adr_name in ovr_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, binary_mean_auc[i], ovr_mean_auc[ovr_adr_names.index(adr_name)], pos_counts[pos_count_index]))
	ofile.close()
	
	#EXTRA TREES VS PAPER
	ofile  = open('common_extratrees_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Extra Trees Mean AUC', 'Paper Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(extra_trees_adr_names):
		if adr_name in paper_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, extra_tress_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)], pos_counts[pos_count_index]))
	ofile.close()
	
	# WEB ALL VS PAPER
	ofile  = open('common_web_paper.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Web Mean AUC', 'Paper Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(web_all_adr_names):
		
		if adr_name in pos_counts_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			pos_count = pos_counts[pos_count_index]
		else:
			pos_count = ""
			
		if adr_name in paper_adr_names:
			writer.writerow((adr_name, web_all_mean_auc[i], paper_mean_auc[paper_adr_names.index(adr_name)], pos_count))
		else:
			writer.writerow((adr_name, web_all_mean_auc[i], "-", pos_count))
	ofile.close()
	
	# TF VS WEB
	ofile  = open('common_tf_web.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Tensorflow Mean AUC', 'Web Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(tf_adr_names):
		if adr_name in web_all_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, tf_mean_auc[i], web_all_mean_auc[web_all_adr_names.index(adr_name)], pos_counts[pos_count_index]))
	
	# OVR VS WEB
	ofile  = open('common_ovr_web.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'OVR Logistic Regression Mean AUC', 'Web Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(ovr_adr_names):
		if adr_name in web_all_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, ovr_mean_auc[i], web_all_mean_auc[web_all_adr_names.index(adr_name)], pos_counts[pos_count_index]))	
	ofile.close()
	
	# BINARY VS WEB
	ofile  = open('common_binary_web.csv', "w")
	writer = csv.writer(ofile, delimiter=',')
	writer.writerow( ('ADR name', 'Binary Logistic Regression Mean AUC', 'Web Mean AUC', 'Positive Count') )
	for i, adr_name in enumerate(binary_adr_names):
		if adr_name in web_all_adr_names:
			pos_count_index = pos_counts_adr_names.index(adr_name)
			writer.writerow((adr_name, binary_mean_auc[i], web_all_mean_auc[web_all_adr_names.index(adr_name)], pos_counts[pos_count_index]))	
	ofile.close()