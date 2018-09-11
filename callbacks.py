import keras
from keras import backend as K


class MomentumScheduler(keras.callbacks.Callback):
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
		mmtm = self.schedule(epoch, self.model.optimizer.momentum)
		assert type(mmtm) == float, 'The output of the "schedule" function should be float.'
		K.set_value(self.model.optimizer.momentum, mmtm)