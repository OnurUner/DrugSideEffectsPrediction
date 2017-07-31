# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest


def save_mean_auc(auc_scores, save_path):
	mean_auc_scores = dict()
	for label_index in auc_scores:
		mean_auc_scores[label_index] = np.mean(auc_scores[label_index])
		
	sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)
	
	f = open(save_path, "w")
	for i in sorted_means:
		f.write(str(i) + " " + str(mean_auc_scores[i]) + "\n")
	f.close()
	
	
def get_train_index(folds, test_index):
	train_index = []
	for i, fold in enumerate(folds):
		if i != test_index:
			train_index += fold
	return train_index


def feature_selection(X, Y, n_best_features=50):
	selected_features = [] 
	for i in xrange(Y.shape[1]):
		y = Y[:,i]
		selector = SelectKBest(chi2, k='all')
		selector.fit(abs(X), y)
		selected_features.append(list(selector.scores_))
	

	feature_scores = np.array(selected_features)
	
	
	avg_scores_per_label = []
	for i in xrange(X.shape[1]):
		avg_scores_per_label.append(np.max(feature_scores[:,i]))
		
	sorted_feature_index = sorted(range(len(avg_scores_per_label)), key=lambda k: avg_scores_per_label[k], reverse=True)
	sorted_feature_scores = sorted(avg_scores_per_label, reverse=True)
	return sorted_feature_index, sorted_feature_scores, X[:, sorted_feature_index[:n_best_features]]