#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from numpy import genfromtxt
sys.path.append('../data/')

from make_dataset import load_dataset
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

current_path = os.path.dirname(os.path.realpath(__file__))
pos_counts_path = current_path + "/../../results/results_filtered/pos_counts.txt"
tf_is_logistic_regression_results_path = current_path + "/../../results/results_filtered/tf_is_logistic_regression_results.txt"
web_results_path = current_path + "/../../results/results_filtered/web_auc.txt"

def save_pos_counts(y, ADRs):
	pos_counts = []
	f = open(pos_counts_path, "w")
	for i in range(y.shape[1]):
		f.write(ADRs[i] + " " + str(np.sum(y[:,i])) + "\n")
		pos_counts.append(np.sum(y[:,i]))
	return pos_counts

def plot(x, y, title, xlabel, ylabel, grid=True):
	plt.figure()
	plt.title(title)
	plt.plot(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(grid)
	plt.show()
	
def pos_count_plot():
	X, y, sample_names, _, ADRs = load_dataset()
	pos_counts = save_pos_counts(y, ADRs)
	plot(range(y.shape[1]), sorted(pos_counts), "Positive Samples Count vs Labels", "label", "positive counts")

def scatter_plot():
	web_adr_names = genfromtxt(web_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()
	tf_adr_names = genfromtxt(tf_is_logistic_regression_results_path, delimiter=" ", usecols=(0), dtype=str).tolist()

	web_mean_auc = genfromtxt(web_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	tf_mean_auc = genfromtxt(tf_is_logistic_regression_results_path, delimiter=" ", usecols=(1), dtype=float).tolist()
	
#	sorted_web_auc_index = sorted(range(len(web_mean_auc)), key=lambda k: web_mean_auc[k], reverse=True)
#	sorted_web_auc = []
#	sorted_adr_names = []
#	y_tf = []
#	y_web = []
#	x_ticks = []
#	
#	for i, adr_name in enumerate(web_adr_names):
#		if adr_name in tf_adr_names:
#			y_tf.append(tf_mean_auc[i])
#			y_web.append(web_mean_auc[web_adr_names.index(adr_name)])
#			x_ticks.append(adr_name)
#	
#	plt.figure()
#	plt.scatter(range(len(x_ticks)), y_tf, c="r", marker='^')
#	plt.scatter(range(len(x_ticks)), y_web, c="g", marker='o')
#	plt.show()
	
	y_tf = []
	y_web = []
	x_ticks = []
	for i, adr_name in enumerate(tf_adr_names):
		if adr_name in web_adr_names:
			y_tf.append(tf_mean_auc[i])
			y_web.append(web_mean_auc[web_adr_names.index(adr_name)])
			x_ticks.append(adr_name)
	plt.figure()
	plt.scatter(range(len(x_ticks)), y_tf, c="r", marker='^')
	plt.scatter(range(len(x_ticks)), y_web, c="g", marker='o')
	plt.show()
	
	print pearsonr(y_tf, y_web)
	
if __name__ == '__main__':
	scatter_plot()
	
	