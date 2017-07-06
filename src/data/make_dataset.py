#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt
from random import shuffle
import os

current_path = os.path.dirname(os.path.realpath(__file__))
input_path = current_path + '/../../data/lincs_sider_gene_expression.csv'
label_path = current_path + '/../../data/lincs_sider_side_effect.csv'

def load_csv(csv_path, delimiter=','):
	row_names = genfromtxt(csv_path, delimiter=delimiter, usecols=(0), dtype=str)
	data = genfromtxt(csv_path, delimiter=delimiter)
	data = data[:,1:]
	return row_names, data

def get_train_test_set(validation_rate=0.3, prune_count=None):
	sample_names, data = load_csv(input_path)
	_, labels = load_csv(label_path)

	sample_count = data.shape[0]
	label_count = labels.shape[1]
		
	sample_indexes = range(sample_count)
	shuffle(sample_indexes)
	validation_count = int(sample_count*validation_rate)
	validation_indexes = sample_indexes[:validation_count]
	training_indexes = sample_indexes[validation_count:]
	
	X_train = data[training_indexes, :]
	X_test = data[validation_indexes, :]
	Y_train = labels[training_indexes, :]
	Y_test = labels[validation_indexes, :]
	
	train_rows = sample_names[training_indexes]
	test_rows = sample_names[validation_indexes]
	
	label_indexes = range(label_count)
	if prune_count is not None:
		label_indexes = []
		for i in range(label_count):
		    if np.sum(Y_train[:,i]) >= prune_count and np.sum(Y_test[:,i]) >= prune_count:
		        label_indexes.append(i)
		
	Y_train = Y_train[:, label_indexes]
	Y_test = Y_test[:, label_indexes]
	
	return X_train, Y_train, train_rows, X_test, Y_test, test_rows, label_indexes

def load_dataset(prune_count=None):
	sample_names, data = load_csv(input_path)
	_, labels = load_csv(label_path)

	sample_count = data.shape[0]
	label_count = labels.shape[1]
		
	sample_indexes = range(sample_count)
	shuffle(sample_indexes)
	
	X = data[sample_indexes, :]
	Y = labels[sample_indexes, :]
	
	label_indexes = range(label_count)
	if prune_count is not None:
		label_indexes = []
		for i in range(label_count):
		    if np.sum(Y[:,i]) >= prune_count:
		        label_indexes.append(i)
		
	Y = Y[:, label_indexes]
	return X, Y, sample_names, label_indexes
