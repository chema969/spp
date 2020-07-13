import tensorflow as tf
import numpy as np
from net_keras import Net
import os
import shutil
import pickle
from losses import qwk_loss, make_cost_matrix, ms_n_qwk_loss
from metrics import np_quadratic_weighted_kappa, top_2_accuracy, top_3_accuracy, \
	minimum_sensitivity, accuracy_off1,categorical_accuracy,mean_absolute_error, \
	overall_mean_squared_error,mean_squared_error
from dataset import Dataset
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import sys


class Experiment:
	"""
	Class that represents a single experiment that can be run and evaluated.
	"""

	def __init__(self, name='unnamed', db='cifar10', net_type='vgg19', batch_size=128, epochs=100,
				 checkpoint_dir='checkpoint', loss='categorical_crossentropy', activation='relu',
				 final_activation='softmax', f_a_params = {}, use_tau=True,
				 prob_layer=None, spp_alpha=1.0, lr=0.1, momentum=0.9, dropout=0, task='both', workers=4,
				 queue_size=1024, rescale_factor=0, augmentation={},
				 val_type='holdout', holdout=0.2, n_folds=5.,encode='one_hot'):
		self._name = name
		self._db = db
		self._net_type = net_type
		self._batch_size = batch_size
		self._epochs = epochs
		self._checkpoint_dir = checkpoint_dir
		self._loss = loss
		self._activation = activation
		self._use_tau = use_tau
		self._final_activation = final_activation
		self._f_a_params = f_a_params
		self._prob_layer = prob_layer
		self._spp_alpha = spp_alpha
		self._lr = lr
		self._momentum = momentum
		self._dropout = dropout
		self._task = task
		self._finished = False
		self._workers = workers
		self._queue_size = queue_size
		self._augmentation = augmentation
		self._val_type = val_type
		self._holdout = holdout
		self._n_folds = n_folds
		self._current_fold = 0
		self._encode=encode		
		self._cost_matrix=None
		self._best_metric = None
		self._optimizer = None
		self._ds = None
		self·_ensemble = False

		# Model and results file names
		self.model_file = 'model'
		self.best_model_file = 'best_model'
		self.model_file_extra = 'model.txt'
		self.csv_file = 'results.csv'
		self.evaluation_file = 'evaluation.pickle'

	def set_auto_name(self):
		"""
		Set experiment name based on experiment parameters.
		:return: None
		"""
		self.name = self.get_auto_name()

	def get_auto_name(self):
		"""
		Get experiment auto-generated name based on experiment parameters.
		:return: experiment auto-generated name.
		"""
		return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.db, self.net_type, self.batch_size, self.activation,
														 self.loss,
														 self.final_activation,
														 self.prob_layer and self.prob_layer or '',
														 self.spp_alpha, self.lr,
														 self.momentum, self.dropout)

	# PROPERTIES

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		self._name = name

	@name.deleter
	def name(self):
		del self._name

	@property
	def db(self):
		return self._db

	@db.setter
	def db(self, db):
		self._db = db

	@db.deleter
	def db(self):
		del self._db

	@property
	def net_type(self):
		return self._net_type

	@net_type.setter
	def net_type(self, net_type):
		self._net_type = net_type

	@net_type.deleter
	def net_type(self):
		del self._net_type

	@property
	def batch_size(self):
		return self._batch_size

	@batch_size.setter
	def batch_size(self, batch_size):
		self._batch_size = batch_size

	@batch_size.deleter
	def batch_size(self):
		del self._batch_size

	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def epochs(self, epochs):
		self._epochs = epochs

	@epochs.deleter
	def epochs(self):
		del self._epochs

	@property
	def checkpoint_dir(self):
		return self._checkpoint_dir

	@checkpoint_dir.setter
	def checkpoint_dir(self, checkpoint_dir):
		self._checkpoint_dir = checkpoint_dir

	@checkpoint_dir.deleter
	def checkpoint_dir(self):
		del self._checkpoint_dir

	@property
	def loss(self):
		return self._loss

	@loss.setter
	def loss(self, loss):
		self._loss = loss

	@loss.deleter
	def loss(self):
		del self._loss

	@property
	def activation(self):
		return self._activation

	@activation.setter
	def activation(self, activation):
		self._activation = activation

	@activation.deleter
	def activation(self):
		del self._activation

	@property
	def final_activation(self):
		return self._final_activation

	@final_activation.setter
	def final_activation(self, final_activation):
		self._final_activation = final_activation

	@final_activation.deleter
	def final_activation(self):
		del self._final_activation

	@property
	def f_a_params(self):
		return self._f_a_params

	@f_a_params.setter
	def f_a_params(self, f_a_params):
		self._f_a_params = f_a_params

	@f_a_params.deleter
	def f_a_params(self):
		del self._f_a_params

	@property
	def use_tau(self):
		return self._use_tau

	@use_tau.setter
	def use_tau(self, use_tau):
		self._use_tau = use_tau

	@use_tau.deleter
	def use_tau(self):
		del self._use_tau

	@property
	def prob_layer(self):
		return self._prob_layer

	@prob_layer.setter
	def prob_layer(self, prob_layer):
		self._prob_layer = prob_layer

	@prob_layer.deleter
	def prob_layer(self):
		del self._prob_layer

	@property
	def spp_alpha(self):
		return self._spp_alpha

	@spp_alpha.setter
	def spp_alpha(self, spp_alpha):
		self._spp_alpha = spp_alpha

	@spp_alpha.deleter
	def spp_alpha(self):
		del self._spp_alpha

	@property
	def lr(self):
		return self._lr

	@lr.setter
	def lr(self, lr):
		self._lr = lr

	@lr.deleter
	def lr(self):
		del self._lr

	@property
	def momentum(self):
		return self._momentum

	@momentum.setter
	def momentum(self, momentum):
		self._momentum = momentum

	@momentum.deleter
	def momentum(self):
		del self._momentum

	@property
	def dropout(self):
		return self._dropout

	@dropout.setter
	def dropout(self, dropout):
		self._dropout = dropout

	@dropout.deleter
	def dropout(self):
		del self._dropout

	@property
	def task(self):
		return self._task

	@task.setter
	def task(self, task):
		self._task = task

	@task.deleter
	def task(self):
		del self._task

	@property
	def finished(self):
		return self._finished

	@finished.setter
	def finished(self, finished):
		self._finished = finished

	@finished.deleter
	def finished(self):
		del self._finished

	@property
	def workers(self):
		return self._workers

	@workers.setter
	def workers(self, workers):
		self._workers = workers

	@workers.deleter
	def workers(self):
		del self._workers

	@property
	def queue_size(self):
		return self._workers

	@queue_size.setter
	def queue_size(self, queue_size):
		self._queue_size = queue_size

	@queue_size.deleter
	def queue_size(self):
		del self._queue_size


	@property
	def augmentation(self):
		return self._augmentation

	@augmentation.setter
	def augmentation(self, augmentation):
		self._augmentation = augmentation

	@augmentation.deleter
	def augmentation(self):
		del self._augmentation

	@property
	def best_metric(self):
		return self._best_metric

	@property
	def current_fold(self):
		return self._current_fold

	@current_fold.setter
	def current_fold(self, current_fold):
		self._current_fold = current_fold

	def new_metric(self, metric, maximize=False):
		"""
		Updates best metric if metric provided is better than the best metric stored.
		:param metric: new metric.
		:param maximize: maximize metric instead of minimizing.
		:return: True if new metric is better than best metric or False otherwise.
		"""
		if self._best_metric is None or (
						maximize and metric > self._best_metric or not maximize and metric <= self._best_metric):
			self._best_metric = metric
			return True
		return False

	# # # # # # #

	def run(self):
		"""
		Run training process.
		:return: None
		"""

		print('=== RUNNING {} ==='.format(self.name))

		# Initial epoch. 0 by default
		start_epoch = 0

		# Load training status
		if os.path.isfile(os.path.join(self.checkpoint_dir, self.model_file_extra)):
			# Continue from the epoch where we were and load the best metric
			with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'r') as f:
				start_epoch = int(f.readline())
				self.new_metric(float(f.readline()))

		if start_epoch >= self.epochs:
			print("Training already finished. Skipping...")
			return


		# Get class weights based on frequency
		class_weight = self._ds.get_class_weights()
		

		# Learning rate scheduler callback

		lr_drop = 20


		def lr_scheduler(epoch):
			return self.lr * (0.5 ** (epoch // lr_drop))


		# Save epoch callback for training process
		def save_epoch(epoch, logs):
			# Check whether new metric is better than best metric
			try:
				logs['val_loss']
			except KeyError:
				# In case validation is not defined (For example, little datasets)
				print("\n\nKey error:Validation loss not defined")
				pass
			else:
				if (self.new_metric(logs['val_loss'])):
					if os.path.isdir(os.path.join(self.checkpoint_dir, self.best_model_file)):					
						shutil.rmtree(os.path.join(self.checkpoint_dir, self.best_model_file))
					tf.keras.models.save_model(model,os.path.join(self.checkpoint_dir, self.best_model_file))
					print("Best model saved.")
			if(epoch%10==0):
				if os.path.isdir(os.path.join(self.checkpoint_dir, self.model_file)):					
					shutil.rmtree(os.path.join(self.checkpoint_dir, self.model_file))
				tf.keras.models.save_model(model,os.path.join(self.checkpoint_dir, self.model_file))
				print("Model saved.")
				with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'w') as f:
					f.write(str(epoch + 1))
					f.write('\n' + str(self.best_metric))


		save_epoch_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_epoch)


		# Create checkpoint dir if not exists
		if not os.path.isdir(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		# Create the cost matrix that will be used to compute qwk
		self._cost_matrix = K.constant(make_cost_matrix(self._ds.num_classes), dtype=K.floatx())

		# Cross-entropy loss by default
		loss = 'categorical_crossentropy'
		
		# Quadratic Weighted Kappa loss
		if self.loss == 'qwk':
			loss = qwk_loss(self._cost_matrix)
		elif self.loss == 'msqwk':
			loss = ms_n_qwk_loss(self._cost_matrix)
		if self._ensemble:
			if self._ensemble_type=='doel3':
				loss =tf.keras.losses.CategoricalCrossentropy(from_logits=True)
			else:
				loss =tf.keras.losses.BinaryCrossentropy(from_logits=True)
		# Check whether a saved model exists
		if os.path.isdir(os.path.join(self.checkpoint_dir, self.model_file)):
			print("===== RESTORING SAVED MODEL =====")
			if loss != 'categorical_crossentropy':
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.model_file),custom_objects={loss.__name__: loss})
			else:
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.model_file))
		elif os.path.isdir(os.path.join(self.checkpoint_dir, self.best_model_file)):
			print("===== RESTORING SAVED BEST MODEL =====")
			if loss != 'categorical_crossentropy':
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.best_model_file),custom_objects={loss.__name__: loss})
			else:
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.best_model_file))
		else:
			# NNet object
			net_object = Net(self._ds.img_size, self.activation, self.final_activation, self.f_a_params, self.use_tau,
						 self.prob_layer, self._ds.num_channels, self._ds.num_classes, self.spp_alpha, self.dropout,self._ensemble,self._ensemble_type)

			model = net_object.build(self.net_type)



		# Only accuracy for training.
		# Computing QWK for training properly is too expensive
		metrics = ['accuracy']

		lr_decay = 1e-6
		opt=tf.keras.optimizers.SGD(lr=self.lr, decay=lr_decay, momentum=self._momentum, nesterov=True)
		if self._optimizer=='Nadam':
			opt=tf.keras.optimizers.SGD(lr=self.lr, decay=lr_decay)
		elif self._optimizer=='Adam':
			opt=tf.keras.optimizers.Adam(lr=self.lr, decay=lr_decay)
		elif self._optimizer=='RMSprop':
			opt=tf.keras.optimizers.RMSprop(lr=self.lr, decay=lr_decay)
		elif self._optimizer=='Adagrad':
			opt=tf.keras.optimizers.Adagrad(lr=self.lr, decay=lr_decay)
		elif self._optimizer=='Adadelta':
			opt=tf.keras.optimizers.Adadelta(lr=self.lr, decay=lr_decay)
		# Compile the keras model
		model.compile(
			optimizer =opt,
			loss=loss, metrics=metrics
		)

		# Print model summary
		model.summary()


		print('Training on {self._ds.size_train()} samples, validating on {self._ds.size_val()} samples.')
		if not self._ensemble:
			# Run training
			model.fit(self._ds.generate_train(self.batch_size, self.augmentation,self._encode,self._soft_ordinal_config), epochs=self.epochs,
							initial_epoch=start_epoch,
							steps_per_epoch=self._ds.num_batches_train(self.batch_size),
							callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
									   save_epoch_callback,
									   tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, self.csv_file),
																	append=True),
									   tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=40, verbose=1)
									   ],
							workers=self.workers,
							use_multiprocessing=False,
							max_queue_size=self.queue_size,
							class_weight=class_weight,
							validation_data=self._ds.generate_val(self.batch_size,self._encode,self._soft_ordinal_config),
							validation_steps=self._ds.num_batches_val(self.batch_size),
							verbose=2
							)


			self.finished = True
		else:
			# Run training
			model.fit(self._ds.generate_train(self.batch_size, self.augmentation,self._encode,self._soft_ordinal_config, ensemble=True, ensemble_train=True, ensemble_type=self._ensemble_type), epochs=self.epochs,
							initial_epoch=start_epoch,
							steps_per_epoch=self._ds.num_batches_train(self.batch_size),
							callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
									   save_epoch_callback,
									   tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, self.csv_file),
																	append=True),
									   tf.keras.callbacks.EarlyStopping(min_delta=0.0005, patience=40, verbose=1)
									   ],
							workers=self.workers,
							use_multiprocessing=False,
							max_queue_size=self.queue_size,
							validation_data=self._ds.generate_val(self.batch_size,self._encode,self._soft_ordinal_config, ensemble=True, ensemble_type=self._ensemble_type),
							validation_steps=self._ds.num_batches_val(self.batch_size),
							verbose=2
							)


			self.finished = True
		# Mark the training as finished in the checkpoint file
		with open(os.path.join(self.checkpoint_dir, self.model_file_extra), 'w') as f:
			f.write(str(self.epochs))
			f.write('\n' + str(self.best_metric))

		# Delete model file
		if os.path.isdir(os.path.join(self.checkpoint_dir, self.model_file)):
			shutil.rmtree(os.path.join(self.checkpoint_dir, self.model_file))


	def evaluate(self):
		"""
		Run evaluation on test data.
		:return: None
		"""
		print('=== EVALUATING {} ==='.format(self.name))

		# Check if best model file exists
		if not os.path.isdir(os.path.join(self.checkpoint_dir, self.best_model_file)):
			print('Best model file not found')
			return

		# Check if model was already evaluated
		if os.path.isfile(os.path.join(self.checkpoint_dir, self.evaluation_file)):
			print('Model already evaluated')
			return

		all_metrics = {}
		import sys
		np.set_printoptions(threshold=sys.maxsize)
		# Get the generators for train, validation and test
		generators = [self._ds.generate_train(self.batch_size, {} ,self._encode, self._soft_ordinal_config,self._ensemble,ensemble_type=self._ensemble_type), 
		self._ds.generate_val(self.batch_size,self._encode,self._soft_ordinal_config,self._ensemble,ensemble_type=self._ensemble_type), 
		self._ds.generate_test(self.batch_size,self._encode,self._soft_ordinal_config,self._ensemble,ensemble_type=self._ensemble_type)]
		steps = [self._ds.num_batches_train(self.batch_size), self._ds.num_batches_val(self.batch_size), self._ds.num_batches_test(self.batch_size)]

		for generator, step, set in zip(generators, steps, ['Train', 'Validation', 'Test']):
			print('\n=== {} dataset ===\n'.format(set))
			if self._cost_matrix==None:
				self._cost_matrix = K.constant(make_cost_matrix(self._ds.num_classes), dtype=K.floatx())

			# Load model
			if self._loss=='qwk':
				loss= qwk_loss(self._cost_matrix)
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.best_model_file),custom_objects={loss.__name__: loss})
			elif self._loss=='msqwk':
				loss= ms_n_qwk_loss(self._cost_matrix)
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.best_model_file),custom_objects={loss.__name__: loss})
			else:
				model=tf.keras.models.load_model(os.path.join(self.checkpoint_dir, self.best_model_file))

			# Get predictions
			predictions = model.predict(generator, steps=step, verbose=1)

			y_set = None
			if not self._ensemble:
				for x, y in generator:
					y_set = np.array(y) if y_set is None else np.vstack((y_set, y))
				metrics = self.compute_metrics(y_set, predictions, self._ds.num_classes)

			else:
				for x, y in generator:
					y_set = y if y_set is None else [np.vstack((j, k)) for j,k in zip(y_set,y)]
				
				if self._ensemble_type=='doel3':
					def create_mat3(y,size):
						u=np.triu(np.ones([size,size]),1)
						i=np.identity(size)
						o=np.tril(np.ones([size,size]),-1)
						ou=np.array([o,i,u])
						prediction=None
						for j in range(y[0].size):
							matrix =[ou[int(y[i][j])][i] for i in range(size)]
							prediction=np.array(np.argmax(np.sum(matrix,0))) if prediction is None else np.hstack((prediction,np.argmax(np.sum(matrix,0))))
						return prediction

					predict=[np.argmax(ar,axis=1) for ar in predictions]
					y_set_aux=[np.argmax(ar,axis=1) for ar in y_set]

					ens_predictions=tf.keras.utils.to_categorical(create_mat3(predict,self._ds.num_classes),num_classes=self._ds.num_classes)
					ens_y=tf.keras.utils.to_categorical(create_mat3(y_set_aux,self._ds.num_classes),num_classes=self._ds.num_classes)

				else:
					"""Firstly, we transform the array of predictions """ 
					predict=[np.squeeze(np.transpose(np.round(ar))).astype(int) for ar in predictions]
					y_set_aux=[np.squeeze(np.transpose(ar)).astype(int) for ar in y_set]
					
					if self._ensemble_type=='doel2':
						def create_mat(y,size):
							u=np.triu(np.ones([size-1,size]),1)
							o=np.tril(np.ones([size-1,size]))
							ou=np.array([o,u])
							prediction=None
							for j in range(y[0].size):
								matrix =[ou[y[i][j]][i] for i in range(size-1)]
								prediction=np.array(np.argmax(np.sum(matrix,0))) if prediction is None else np.hstack((prediction,np.argmax(np.sum(matrix,0))))
							return np.array(prediction)

						ens_predictions=tf.keras.utils.to_categorical(create_mat(predict,self._ds.num_classes),num_classes=self._ds.num_classes)
						ens_y=tf.keras.utils.to_categorical(np.sum(y_set_aux,axis=0),num_classes=self._ds.num_classes)

					else:
						ens_predictions=tf.keras.utils.to_categorical(np.sum(predict,axis=0),num_classes=self._ds.num_classes)
						ens_y=tf.keras.utils.to_categorical(np.sum(y_set_aux,axis=0),num_classes=self._ds.num_classes)
				metrics = self.compute_metrics(ens_y,ens_predictions,self._ds.num_classes)

			self.print_metrics(metrics)

			all_metrics[set] = metrics

		with open(os.path.join(self.checkpoint_dir, self.evaluation_file), 'wb') as f:
			pickle.dump({'config': self.get_config(), 'metrics': all_metrics}, f)


	def compute_metrics(self, y_true, y_pred, num_classes):
		# Calculate metric
		qwk = np_quadratic_weighted_kappa(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), 0,
										  num_classes - 1)
										  		
		ms = minimum_sensitivity(y_true, y_pred)
		mae = mean_absolute_error(y_true, y_pred)
		omae = overall_mean_squared_error(y_true, y_pred)
		mse = mean_squared_error(y_true, y_pred)
		acc = categorical_accuracy(y_true, y_pred)
		top2 = top_2_accuracy(y_true, y_pred)
		top3 = top_3_accuracy(y_true, y_pred)
		off1 = accuracy_off1(y_true, y_pred)
		conf_mat = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

		metrics = {
			'QWK': qwk,
			'MS': ms,
			'MAE': mae,
			'OMAE': omae,
			'MSE': mse,
			'CCR': acc,
			'Top-2': top2,
			'Top-3': top3,
			'1-off': off1,
			'Confusion matrix': conf_mat
		}

		return metrics

	def print_metrics(self, metrics):
		print('Confusion matrix :\n{}'.format(metrics['Confusion matrix']))
		print('QWK: {:.4f}'.format(metrics['QWK']))
		print('CCR: {:.4f}'.format(metrics['CCR']))
		print('Top-2: {:.4f}'.format(metrics['Top-2']))
		print('Top-3: {:.4f}'.format(metrics['Top-3']))
		print('1-off: {:.4f}'.format(metrics['1-off']))
		print('MAE: {:.4f}'.format(metrics['MAE']))
		print('OMAE: {:.4f}'.format(metrics['OMAE']))
		print('MSE: {:.4f}'.format(metrics['MSE']))
		print('MS: {:.4f}'.format(metrics['MS']))


	def get_config(self):
		"""
		Get config dictionary from object config.
		:return: config dictionary.
		"""
		return {
			'name': self.name,
			'db': self.db,
			'net_type': self.net_type,
			'batch_size': self.batch_size,
			'epochs': self.epochs,
			'checkpoint_dir': self.checkpoint_dir,
			'prob_layer': self.prob_layer,
			'loss': self.loss,
			'activation': self.activation,
			'use_tau' : self.use_tau,
			'final_activation': self.final_activation,
			'f_a_params': self.f_a_params,
			'spp_alpha': self.spp_alpha,
			'lr': self.lr,
			'momentum': self.momentum,
			'dropout': self.dropout,
			'task': self.task,
			'workers': self.workers,
			'queue_size': self.queue_size,
			'augmentation': self.augmentation,
			'val_type' : self._val_type,
			'holdout' : self._holdout,
			'n_folds' : self._n_folds,
			'optimizer'  : self._optimizer,
			'encode'  : self._encode,
			'soft_ordinal_config'  : self._soft_ordinal_config,
			'ensemble' : self._ensemble,
			'ensemble_type' : self._ensemble_type
		}

	def set_config(self, config):
		"""
		Set object config from config dictionary
		:param config: config dictionary.
		:return: None
		"""
		self.db = 'db' in config and config['db'] or 'cifar10'
		self.net_type = 'net_type' in config and config['net_type'] or 'vgg19'
		self.batch_size = 'batch_size' in config and int(config['batch_size']) or 128
		self.epochs = 'epochs' in config and config['epochs'] or 100
		self.checkpoint_dir = 'checkpoint_dir' in config and config['checkpoint_dir'] or 'results'
		self.loss = 'loss' in config and config['loss'] or 'categorical_crossentropy'
		self.activation = 'activation' in config and config['activation'] or 'relu'
		self.final_activation = 'final_activation' in config and config['final_activation'] or 'softmax'
		self.f_a_params = config['f_a_params'] if 'f_a_params' in config else {}
		self.use_tau = config['use_tau'] if 'use_tau' in config and config['use_tau'] else False
		self.prob_layer = 'prob_layer' in config and config['prob_layer'] or None
		self.spp_alpha = 'spp_alpha' in config and config['spp_alpha'] or 0
		self.lr = 'lr' in config and config['lr'] or 0.1
		self.momentum = 'momentum' in config and config['momentum'] or 0.9
		self.dropout = 'dropout' in config and config['dropout'] or 0
		self.task = 'task' in config and config['task'] or 'both'
		self.workers = 'workers' in config and config['workers'] or 4
		self.queue_size = 'queue_size' in config and config['queue_size'] or 1024
		self.augmentation = 'augmentation' in config and config['augmentation'] or {}
		self._val_type = 'val_type' in config and config['val_type'] or 'holdout'
		self._holdout = 'holdout' in config and float(config['holdout']) or 0.2
		self._n_folds = 'n_folds' in config and int(config['n_folds']) or 5
		self._optimizer = 'optimizer' in config and config['optimizer'] or 'SGD'
		self._encode = 'encode' in config and config['encode'] or 'one_hot'
		self._soft_ordinal_config = 'soft_ordinal_config' in config and config['soft_ordinal_config'] or 'absolute'
		self._ensemble = config['ensemble'] if 'ensemble' in config and config['ensemble'] else False
		self._ensemble_type = config['ensemble_type'] if 'ensemble_type' in config and config['ensemble_type'] else 'regression'
		if 'name' in config:
			self.name = config['name']
		else:
			self.set_auto_name()

		
		# Load dataset
		self._ds = Dataset(self._db)

		self._setup_validation()

	def _setup_validation(self):
		if self._ds is None:
			raise Exception('Cannot setup validation because dataset is not loaded')
			
		# Validation config
		if self._val_type == 'kfold':
			self._ds.n_folds = self._n_folds
			self._ds.set_fold(self._current_fold)
		elif self._val_type == 'holdout':
			self._ds.n_folds = 1 # 1 fold means holdout
			self._ds.holdout = self._holdout
		else:
			raise Exception('{} is not a valid validation type.'.format(self._val_type))

	def save_to_file(self, path):
		"""
		Save experiment to pickle file.
		:param path: path where pickle file will be saved.
		:return: None
		"""
		pickle.dump(self.get_config(), path)

	def load_from_file(self, path):
		"""
		Load experiment from pickle file.
		:param path: path where pickle file is located.
		:return: None
		"""
		if os.path.isfile(path):
			self.set_config(pickle.load(path))
