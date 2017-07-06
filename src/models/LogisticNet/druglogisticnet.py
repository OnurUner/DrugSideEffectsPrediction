#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append('./../')

import os
import numpy as np
from keras.models import Model 
from keras.layers import Dense, Input
from base_deepnet import BaseDeepNet
from keras.regularizers import l1_l2
from sklearn.utils.class_weight import compute_class_weight

current_path = os.path.dirname(os.path.realpath(__file__))
weights_path = current_path + "/weights/druglogisticnet_weights.npy"

class DrugLogisticNet(BaseDeepNet):
	
	def __init__(self, input_shape, output_shape, output_names):
		self.input_shape = input_shape
		self.task_count = output_shape
		self.weights_path = weights_path
		self.output_names = output_names
	
	def create_model(self):
		self.outputs = []
		inputs = Input(shape=self.input_shape)
		reg = l1_l2(l1=0.01, l2=0.01)
		for i in range(self.task_count):
			self.outputs.append(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', kernel_regularizer=reg , bias_initializer='ones', name=self.output_names[i])(inputs))
		
		self.model = Model(inputs=inputs, outputs=self.outputs)
		self.model.compile(optimizer='adam',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])
		return self.model
		
	def fit(self, x_train, y_train_dict, epochs=30, batch_size=30, is_balanced=False, verbose=0):
		self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
		if is_balanced:
			weigths = self.calc_class_weights(y_train_dict)
			self.model.fit(x_train, y_train_dict, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.3, class_weight=weigths)
		else:
			self.model.fit(x_train, y_train_dict, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.3)
		return self.model
	
	def calc_class_weights(self, y_train_dict):
		class_weights = dict()
		for out_name in y_train_dict:
			weights = dict()
			counts = np.bincount(y_train_dict[out_name].astype(int))
			weights[0] = 1.0/counts[0]
			weights[1] = 1.0/counts[1]
			class_weights[out_name] = weights
		return class_weights
		