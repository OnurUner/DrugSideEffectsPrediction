#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
sys.path.append('../data/')

from make_dataset import load_dataset
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(__file__))
pos_counts_path = current_path + '/../../results/pos_counts.txt'

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

if __name__ == '__main__':
	X, y, sample_names, _, ADRs = load_dataset()
	pos_counts = save_pos_counts(y, ADRs)
	plot(range(y.shape[1]), sorted(pos_counts), "Positive Samples Count vs Labels", "label", "positive counts")
	