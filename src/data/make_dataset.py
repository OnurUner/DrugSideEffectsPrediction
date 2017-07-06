#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from numpy import genfromtxt
from random import shuffle
import os

current_path = os.path.dirname(os.path.realpath(__file__))
input_path = current_path + '/../../data/lincs_sider_gene_expression.csv'
#label_path = current_path + '/../../data/lincs_sider_side_effect.csv'
label_path = current_path + '/../../data/SIDER_PTs.csv'

def load_data_csv(csv_path, delimiter=','):
	row_names = genfromtxt(csv_path, delimiter=delimiter, usecols=(0), dtype=str)
	data = genfromtxt(csv_path, delimiter=delimiter)
	data = data[:,1:]
	return row_names.tolist(), data

def load_label_csv(csv_path, delimiter=','):
	row_names = genfromtxt(csv_path, delimiter=delimiter, usecols=(0), skip_header=1, dtype=str)
	labels = genfromtxt(csv_path, delimiter=delimiter, usecols=range(1, 3166), skip_header=1)
	side_effect_names = genfromtxt(csv_path, delimiter=delimiter, usecols=range(1, 3166), max_rows=1, dtype=str)
	for i, name in enumerate(side_effect_names):
		side_effect_names[i] = name.replace(' ', '_')
	return row_names.tolist(), labels, side_effect_names.tolist()

def get_train_test_set(validation_rate=0.3, prune_count=None):
	drug_names_data, data = load_data_csv(input_path)
	drug_names_label, labels, ADRs = load_label_csv(label_path)
	
	label_row_indexes = []
	data_row_indexes = []
	for i, drug_data in enumerate(drug_names_data):
		if drug_data in drug_names_label:
			label_row_indexes.append(drug_names_label.index(drug_data))
			data_row_indexes.append(i)
	
	data = data[data_row_indexes,:]
	labels = labels[label_row_indexes,:]
		
	sample_count = data.shape[0]
	sample_indexes = range(sample_count)
	shuffle(sample_indexes)
	validation_count = int(sample_count*validation_rate)
	validation_indexes = sample_indexes[:validation_count]
	training_indexes = sample_indexes[validation_count:]
	
	X_train = data[training_indexes, :]
	X_test = data[validation_indexes, :]
	Y_train = labels[training_indexes, :]
	Y_test = labels[validation_indexes, :]
	
	train_rows = np.array(drug_names_data)[training_indexes]
	test_rows = np.array(drug_names_data)[validation_indexes]
	
	label_indexes = range(labels.shape[1])
	if prune_count is not None:
		label_indexes = []
		for i in range(labels.shape[1]):
		    if np.sum(Y_train[:,i]) >= prune_count and np.sum(Y_test[:,i]) >= prune_count:
		        label_indexes.append(i)
		
	Y_train = Y_train[:, label_indexes]
	Y_test = Y_test[:, label_indexes]
	ADRs = np.array(ADRs)[label_indexes]
	
	return X_train, Y_train, train_rows, X_test, Y_test, test_rows, label_indexes, ADRs 

def load_dataset(prune_count=None):
	drug_names_data, data = load_data_csv(input_path)
	drug_names_label, labels, ADRs = load_label_csv(label_path)
	
	label_row_indexes = []
	data_row_indexes = []
	for i, drug_data in enumerate(drug_names_data):
		if drug_data in drug_names_label:
			label_row_indexes.append(drug_names_label.index(drug_data))
			data_row_indexes.append(i)
	
	X = data[data_row_indexes, :]
	Y = labels[label_row_indexes, :]
	
	label_indexes = range(labels.shape[1])
	if prune_count is not None:
		label_indexes = []
		for i in range(labels.shape[1]):
		    if np.sum(Y[:,i]) >= prune_count:
		        label_indexes.append(i)
		
	Y = Y[:, label_indexes]
	ADRs = np.array(ADRs)[label_indexes]
	return X, Y, drug_names_data, label_indexes, ADRs

#if __name__ == '__main__':
#	drug_names_data, data = load_data_csv(input_path)
#	drug_names_label, labels, ADRs = load_label_csv(label_path)
#	
#	label_row_indexes = []
#	data_row_indexes = []
#	for i, drug_data in enumerate(drug_names_data):
#		if drug_data in drug_names_label:
#			label_row_indexes.append(drug_names_label.index(drug_data))
#			data_row_indexes.append(i)

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	