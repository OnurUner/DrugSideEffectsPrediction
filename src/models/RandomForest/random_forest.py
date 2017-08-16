# -*- coding: utf-8 -*-

import sys
import numpy as np
import os
sys.path.append('../../data/')
sys.path.append('../../utils/')
sys.path.append('../../feature_selection/')

from make_dataset import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from utility import save_mean_auc, prob_to_pred, find_optimal_cutoff
from select_feature import feature_selection

current_path = os.path.dirname(os.path.realpath(__file__))
extra_trees_result_path = current_path + '/../../../results/results_filtered/random_forest_results_baseline.txt'

	
def list_to_nparray(y):
	t_y = y
	for i, column in enumerate(t_y):
		t_y[i] = column[:,1]
	return np.column_stack(t_y)

if __name__ == '__main__':
	feature_index = feature_selection(method="forest", top=978)
	X, Y, sample_names, _, ADRs = load_dataset(prune_count=11)
	
	kf = StratifiedKFold(n_splits=3)
	
	auc_scores = dict()
	for selected_label in xrange(0,Y.shape[1]):
		if selected_label not in feature_index:
			print "Skip label", selected_label
			continue
		
		print "Label", selected_label
		y = Y[:,selected_label]
		x = X[:,feature_index[selected_label]]
		
		for train_index, test_index in kf.split(x, y):
			x_train = x[train_index]
			y_train = y[train_index]
			
			x_test = x[test_index]
			y_test = y[test_index]
				
			clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
			clf.fit(x_train, y_train)
			y_probs = clf.predict_proba(x_test)[:,0]
						
			score = roc_auc_score(y_test, y_probs, average="micro")


			side_effect = ADRs[selected_label]
			if side_effect not in auc_scores:	
				auc_scores[side_effect] = []
			auc_scores[side_effect].append(score)
			
	save_mean_auc(auc_scores, extra_trees_result_path)