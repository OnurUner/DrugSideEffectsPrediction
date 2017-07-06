# -*- coding: utf-8 -*-
import sys
sys.path.append('./../')

import os
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import TensorBoard, CSVLogger
from base_deepnet import BaseDeepNet

current_path = os.path.dirname(os.path.realpath(__file__))
weights_path = current_path + "/weights/drugnet_weights.npy"
csv_log_path = current_path + "/../../log/csv_log/training.log"
tensorboard_log_path = current_path + "/../../log/tensorboard_log"

class DrugNet(BaseDeepNet):
	def __init__(self, input_shape, task_count):
		self.input_shape = input_shape
		self.task_count = task_count
		self.weights_path = weights_path
	
	def create_model(self):
		self.outputs = []
		inputs = Input(shape=self.input_shape)
		x = Dense(20, activation='relu', kernel_initializer='glorot_normal', bias_initializer='ones')(inputs)
		x = Dropout(0.6)(x)
		x = Dense(10, activation='relu', kernel_initializer='glorot_normal', bias_initializer='ones')(x)
		x = Dropout(0.5)(x)
		for i in range(self.task_count):
			self.outputs.append(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='ones')(x))
		
		self.model = Model(inputs=inputs, outputs=self.outputs)
		self.model.compile(optimizer='adam',
			  loss='binary_crossentropy',
			  metrics=['accuracy'])
		return self.model
	
	def fit(self, x, y, epochs = 30, batch_size = 30, verbose=0):
		csv_logger = CSVLogger(csv_log_path, append=True)
		tensorboard = TensorBoard(log_dir=tensorboard_log_path, histogram_freq=0, write_graph=True, write_images=True)
		callbacks_list = [tensorboard, csv_logger]
		self.model.fit(x, y, validation_split=0.33, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks_list)
		return self.model