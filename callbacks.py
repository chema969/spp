import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from metrics import quadratic_weighted_kappa_cm
from losses import make_cost_matrix

class MomentumScheduler(tf.keras.callbacks.Callback):
	'''Momentum scheduler.
	# Arguments
	schedule: a function that takes an epoch index (integer, indexed from 0) and current momentum as input
	and returns a new momentum as output (float).
	'''
	def __init__(self, schedule):
		super(MomentumScheduler, self).__init__()
		self.schedule = schedule

	def on_epoch_begin(self, epoch, logs={}):
		assert hasattr(self.model.optimizer, 'momentum'), \
		'Optimizer must have a "momentum" attribute.'
		mmtm = self.schedule(epoch)
		assert type(mmtm) == float, 'The output of the "schedule" function should be float.'
		tf.assign(self.model.optimizer.momentum, mmtm)


class ValidationCallback(tf.keras.callbacks.Callback):

	def __init__(self, val_generator, num_classes):
		self.val_generator = val_generator
		self.classes = []
		self.num_classes = num_classes
		self.cost_matrix = make_cost_matrix(self.num_classes)

		for i in range(0, num_classes):
			self.classes.append(i)


	def on_epoch_end(self, epoch, logs={}):
		sess = tf.keras.backend.get_session()
		conf_mat = None
		mean_acc = 0
		mean_loss = 0
		batch_count = 0

		for x, y in self.val_generator:
			if batch_count >= 300:
				break

			prediction = self.model.predict_on_batch(x)
			loss = self.model.test_on_batch(x, y)[0]
			y = np.argmax(y, axis=1)
			prediction = np.argmax(prediction, axis=1)

			if conf_mat is None:
				conf_mat = confusion_matrix(y, prediction, labels=self.classes)
			else:
				conf_mat += confusion_matrix(y, prediction, labels=self.classes)

			batch_count += 1
			mean_acc += accuracy_score(y, prediction)
			mean_loss += loss



		mean_acc /= batch_count
		mean_loss /= batch_count
		qwk = sess.run(quadratic_weighted_kappa_cm(conf_mat, self.num_classes, self.cost_matrix))
		logs['val_acc'] = mean_acc
		logs['val_qwk'] = qwk
		logs['val_loss'] = mean_loss

		print('\nval_loss: {} - val_acc: {} - val_qwk: {}'.format(mean_loss, mean_acc, qwk))
		print(conf_mat)