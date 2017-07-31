# -*- coding: utf-8 -*-

import cPickle as pickle
from skmultilearn.dataset import load_dataset_dump
from make_dataset import load_dataset

from scipy import sparse
import numpy as np
from scipy.sparse import lil_matrix
import os

current_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = current_path + '/../../data/folds_prune11.p'


def pos_samples_per_label(Y):
	y = sparse.csr_matrix(Y)
	y_train = lil_matrix(y)
	samples_with_label = [[] for i in range(Y.shape[1])]

	for sample, labels in enumerate(y_train.rows):
		for label in labels:
			samples_with_label[label].append(sample)
	return samples_with_label

def pos_samples_per_labelpair(Y):
	y = sparse.csr_matrix(Y)
	y_train = lil_matrix(y)
	samples_with_labelpairs = {}
	for row, labels in enumerate(y_train.rows):
		pairs = [(a, b) for b in labels for a in labels if a <= b]
		for p in pairs:
			if p not in samples_with_labelpairs:
				samples_with_labelpairs[p] = []
			samples_with_labelpairs[p].append(row)
	return samples_with_labelpairs


# Label Distribution
def LD(folds, y):
	label_count = y.shape[1]
	samples_with_label = pos_samples_per_label(y)
	D = y.shape[0]
	ld = 0.0
	for i in xrange(label_count):
		x = 0.0
		for j in xrange(len(folds)):
			fold = folds[j]
			DI = len(samples_with_label[i])
			Sj = len(fold)
			SIj = len(set(samples_with_label[i]).intersection(fold))
			
			x += abs((SIj/(Sj-SIj)) - (DI/(D-DI)))
		ld += x/len(folds)
		
	return ld/label_count
	

# Label Pair Distribution
def LPD(folds, y):
	samples_with_labelpairs = pos_samples_per_labelpair(y)

	D = y.shape[0]
	ld = 0.0
	for i in samples_with_labelpairs.keys():
		x = 0.0
		for j in xrange(len(folds)):
			fold = folds[j]
			DI = len(samples_with_labelpairs[i])
			Sj = len(fold)
			SIj = len(set(samples_with_labelpairs[i]).intersection(fold))
			
			x += abs((SIj/(Sj-SIj)) - (DI/(D-DI)))
		ld += x/len(folds)
		
	return ld/len(samples_with_labelpairs.keys())
	

# the number of folds that contain at least one label with no positive examples
def FZ(folds, y):
	label_samples = pos_samples_per_label(y)
	n_fold = 0
	for fold in folds:
		label_with_pos_sample = [0 for i in range(y.shape[1])]
		for sample in fold:
			for i, pos_samples in enumerate(label_samples):
				if label_with_pos_sample[i] == 0:
					if sample in pos_samples:
						label_with_pos_sample[i] = 1
		if y.shape[1] != np.sum(label_with_pos_sample):
			n_fold += 1 
	return n_fold
	

# the number of fold-label pairs with no positive examples
def FLZ(folds, y):
	label_samples = pos_samples_per_label(y)
	n_fold_label_pair = 0
	for fold in folds:
		label_with_pos_sample = [0 for i in range(y.shape[1])]
		for sample in fold:
			for i, pos_samples in enumerate(label_samples):
				if label_with_pos_sample[i] == 0:
					if sample in pos_samples:
						label_with_pos_sample[i] = 1
		n_fold_label_pair  += y.shape[1] - np.sum(label_with_pos_sample)
	return n_fold_label_pair 


# the number of fold - label pair pairs with no positive examples
def FLPZ(folds, y):
	labelpair_samples = pos_samples_per_labelpair(y)
	n_fold_labelpair_pair = 0
	for fold in folds:
		labelpair_with_pos_sample = {key:False for key in labelpair_samples.keys()}
		for sample in fold:
			for key in labelpair_samples.keys():
				if not labelpair_with_pos_sample[key]:
					if sample in labelpair_samples[key]:
						labelpair_with_pos_sample[key] = True
		n_labelpair_with_pos = 0
		for key in labelpair_with_pos_sample.keys():
			if labelpair_with_pos_sample[key]:
				n_labelpair_with_pos += 1
		n_fold_labelpair_pair += len(labelpair_samples.keys()) - n_labelpair_with_pos
	return n_fold_labelpair_pair

	
def is_fold(n_splits, y):
	y_train = lil_matrix(y)
	n_samples = y_train.shape[0]
	n_labels = y_train.shape[1]
	percentage_per_fold = [1/float(n_splits) for i in range(n_splits)]
	desired_samples_per_fold = np.array([percentage_per_fold[i]*n_samples for i in range(n_splits)])

	folds = [[] for i in range(n_splits)]

	samples_with_label = [[] for i in range(n_labels)]

	for sample, labels in enumerate(y_train.rows):
		for label in labels:
			samples_with_label[label].append(sample)

	desired_samples_per_label_per_fold = {i: [len(samples_with_label[i])*percentage_per_fold[j] for j in range(n_splits)] for i in range(n_labels)}

	rows_used = {i : False for i in range(n_samples)}
	labeled_samples_available = map(len, samples_with_label)
	total_labeled_samples_available = sum(labeled_samples_available)
	while total_labeled_samples_available > 0:
		l = np.argmin(np.ma.masked_equal(labeled_samples_available, 0, copy=False))

		while len(samples_with_label[l])>0:
			row = samples_with_label[l].pop()
			if rows_used[row]:
				continue

			max_val = max(desired_samples_per_label_per_fold[l])
			M = np.where(np.array(desired_samples_per_label_per_fold[l])==max_val)[0]
			m = None
			if len(M) == 1:
				m = M[0]
			else:
				max_val = max(desired_samples_per_fold[M])
				M_prim = np.where(np.array(desired_samples_per_fold)==max_val)[0]
				M_prim = np.array([x for x in M_prim if x in M])
				m = np.random.choice(M_prim, 1)[0]

			folds[m].append(row)
			rows_used[row]=True
			for i in y_train.rows[row]:
				desired_samples_per_label_per_fold[i][m]-=1
			desired_samples_per_fold[m]-=1

		labeled_samples_available = map(len, samples_with_label)
		total_labeled_samples_available = sum(labeled_samples_available)

	available_samples = [i for i, v in rows_used.iteritems() if not v]
	samples_left = len(available_samples)

	assert (samples_left + sum(map(len, folds))) == n_samples

	while samples_left>0:
		row = available_samples.pop()
		rows_used[row]=True
		fold_selected = np.random.choice(np.where(desired_samples_per_fold>0)[0], 1)[0]
		folds[fold_selected].append(row)
		samples_left-=1

	assert sum(map(len, folds)) == n_samples
	assert len([i for i, v in rows_used.iteritems() if not v])==0
	return folds


def sois_fold(n_splits, y):
	y_train = lil_matrix(y)
	
	n_samples = y_train.shape[0]
	n_labels = y_train.shape[1]
	
	percentage_per_fold = [1/float(n_splits) for i in range(n_splits)]
	desired_samples_per_fold = np.array([percentage_per_fold[i]*n_samples for i in range(n_splits)])
	
	folds = [[] for i in range(n_splits)]
	
	samples_with_label = [[] for i in range(n_labels)]
	
	for sample, labels in enumerate(y_train.rows):
		for label in labels:
			samples_with_label[label].append(sample)
	
	desired_samples_per_label_per_fold = {i: [len(samples_with_label[i])*percentage_per_fold[j] for j in range(n_splits)] for i in range(n_labels)}
	
	samples_with_labelpairs = {}
	for row, labels in enumerate(y_train.rows):
		pairs = [(a, b) for b in labels for a in labels if a <= b]
		for p in pairs:
			if p not in samples_with_labelpairs:
				samples_with_labelpairs[p] = []
			samples_with_labelpairs[p].append(row)
	
	desired_samples_per_labelpair_per_fold = {k : [len(v)*i for i in percentage_per_fold] for k,v in samples_with_labelpairs.iteritems()}
	
	labels_of_edges = samples_with_labelpairs.keys()
	labeled_samples_available = [len(samples_with_labelpairs[v]) for v in labels_of_edges]
	
	rows_used = {i : False for i in range(n_samples)}
	total_labeled_samples_available = sum(labeled_samples_available)
	
	while total_labeled_samples_available > 0:
		l = labels_of_edges[np.argmin(np.ma.masked_equal(labeled_samples_available, 0, copy=False))]
	
	
		while len(samples_with_labelpairs[l])>0:
	
			row = samples_with_labelpairs[l].pop()
			if rows_used[row]:
				continue
	
			max_val = max(desired_samples_per_labelpair_per_fold[l])
			M = np.where(np.array(desired_samples_per_labelpair_per_fold[l])==max_val)[0]
		#    print l, M, len(M)
	
			m = None
			if len(M) == 1:
				m = M[0]
			else:
				max_val = max(desired_samples_per_fold[M])
				M_bis = np.where(np.array(desired_samples_per_fold)==max_val)[0]
				M_bis = np.array([x for x in M_bis if x in M])
				m = np.random.choice(M_bis, 1)[0]
		#        print M_prim,m, max_val, desired_samples_per_labelpair_per_fold[l]
	
			folds[m].append(row)
			rows_used[row]=True
			desired_samples_per_labelpair_per_fold[l][m]-=1
			if desired_samples_per_labelpair_per_fold[l][m] <0:
				desired_samples_per_labelpair_per_fold[l][m]=0
	
			for i in samples_with_labelpairs.iterkeys():
				if row in samples_with_labelpairs[i]:
					samples_with_labelpairs[i].remove(row)
					desired_samples_per_labelpair_per_fold[i][m]-=1
	
				if desired_samples_per_labelpair_per_fold[i][m] <0:
					desired_samples_per_labelpair_per_fold[i][m]=0
			desired_samples_per_fold[m]-=1
	
		labeled_samples_available = [len(samples_with_labelpairs[v]) for v in labels_of_edges]
		total_labeled_samples_available = sum(labeled_samples_available)
	
		available_samples = [i for i, v in rows_used.iteritems() if not v]
		samples_left = len(available_samples)
	
	labeled_samples_available = map(len, samples_with_label)
	total_labeled_samples_available = sum(labeled_samples_available)
	
	while total_labeled_samples_available > 0:
		l = np.argmin(np.ma.masked_equal(labeled_samples_available, 0, copy=False))
	
		while len(samples_with_label[l])>0:
			row = samples_with_label[l].pop()
			if rows_used[row]:
				continue
	
			max_val = max(desired_samples_per_label_per_fold[l])
			M = np.where(np.array(desired_samples_per_label_per_fold[l])==max_val)[0]
			m = None
			if len(M) == 1:
				m = M[0]
			else:
				max_val = max(desired_samples_per_fold[M])
				M_prim = np.where(np.array(desired_samples_per_fold)==max_val)[0]
				M_prim = np.array([x for x in M_prim if x in M])
				m = np.random.choice(M_prim, 1)[0]
	
			folds[m].append(row)
			rows_used[row]=True
			for i in y_train.rows[row]:
				desired_samples_per_label_per_fold[i][m]-=1
			desired_samples_per_fold[m]-=1
	
		labeled_samples_available = map(len, samples_with_label)
		total_labeled_samples_available = sum(labeled_samples_available)
	
	assert (samples_left + sum(map(len, folds))) == n_samples
	
	while samples_left>0:
		row = available_samples.pop()
		rows_used[row]=True
		fold_selected = np.random.choice(np.where(desired_samples_per_fold>0)[0], 1)[0]
		folds[fold_selected].append(row)
		samples_left-=1
	
	assert sum(map(len, folds)) == n_samples
	assert len([i for i, v in rows_used.iteritems() if not v])==0
	return folds


def create_folds(prune_count):
	X, Y, drug_names_data, label_indexes, ADRs = load_dataset(prune_count=prune_count)
	y = sparse.csr_matrix(Y)
	n_splits = 3
	SOIS = sois_fold(n_splits, y)
	IS = is_fold(n_splits, y)
	
	dataset = {}
	dataset["SOIS"] = SOIS
	dataset["IS"] = IS
	dataset["X"] = X
	dataset["Y"] = Y
	dataset["sample_names"] = drug_names_data
	dataset["label_indexes"] = label_indexes
	dataset["ADRs"] = ADRs
	
	pickle.dump(dataset, open(dataset_path, "wb"))


def load_folds():
	dataset = pickle.load(open(dataset_path, "rb"))
	X = dataset["X"]
	Y = dataset["Y"]
	sample_names = dataset["sample_names"]
	ADRs = dataset["ADRs"]
	SOIS = dataset["SOIS"]
	IS = dataset["IS"]
	
	return X, Y, sample_names, ADRs, SOIS, IS
	

#if __name__ == '__main__':
#	create_folds(prune_count=11)
#	X, Y, sample_names, ADRs, SOIS, IS	= load_folds()
#	
#	sois_ld = LD(SOIS, Y)
#	is_ld = LD(IS, Y)
#	
#	sois_lpd = LPD(SOIS, Y)
#	is_lpd = LPD(IS, Y)
#	
#	sois_fz = FZ(SOIS, Y)
#	is_fz = FZ(IS, Y)
#
#	sois_flz = FLZ(SOIS, Y)
#	is_flz = FLZ(IS, Y)
#	
#	sois_flpz = FLPZ(SOIS, Y)
#	is_flpz = FLPZ(IS, Y)
#	
#	print "      LD      "
#	print "--------------"
#	print "SOIS: %.5f \n  IS: %.5f" % (sois_ld, is_ld)
#	print "----------"
#
#	print "      LPD       "
#	print "----------------"
#	print "SOIS: %.5f \n  IS: %.5f" % (sois_lpd, is_lpd)
#	print "----------------"
#	
#	print "      FZ      "
#	print "--------------"
#	print "SOIS: %.5f \n  IS: %.5f" % (sois_fz, is_fz)
#	print "--------------"
#	
#	print "      FLZ       "
#	print "----------------"
#	print "SOIS: %.5f \n  IS: %.5f" % (sois_flz, is_flz)
#	print "----------------"
#	
#	print "      FLPZ      "
#	print "----------------"
#	print "SOIS: %.5f \n  IS: %.5f" % (sois_flpz, is_flpz)
#	print "----------------"
	
	
	