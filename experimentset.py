import os
import json
import multiprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from experiment import Experiment

class ExperimentSet:
	"""
	Set of experiments that can be executed sequentially.
	"""
	def __init__(self, json_path):
		self._json_path = json_path

	
	def _generate_experiments(self):
		# Load JSON file
		with open(self._json_path) as f:
			configs = json.load(f)

		# Add experiments
		#For each experiment
		for config in configs:
			val_type = config['val_type'] if 'val_type' in config else 'holdout'
			if val_type == 'holdout' and 'executions' in config:
				executions = int(config['executions'])
			elif val_type == 'kfold' and 'n_folds' in config:
				executions = int(config['n_folds'])
			else:
				raise Exception(F"{val_type} is not a valid validation type.")
			
			#For each fold/execution in the experiment
			for execution in range(0, executions):
				exec_config = config.copy()
				if 'name' in exec_config:
					exec_config['name'] += "_{}".format(execution)
				exec_config['checkpoint_dir'] += "/{}".format(execution)
				experiment = Experiment()
				experiment.current_fold = execution
				experiment.set_config(exec_config)
				#Creates a generator with all executions and all experimets
				yield experiment

	def _run_one(self,gpu_number,experiment):
		#Allow soft device placement, so in case the device is not found, TensorFlow will automatically choose an existing and supported device
		tf.config.set_soft_device_placement(True)
		physical_devices = tf.config.experimental.list_physical_devices('GPU')
		try: 
			tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
		except: 
 			# Invalid device or cannot modify virtual devices once initialized. 
			pass 
		os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
		with tf.device('/device:GPU:' + str(gpu_number)):
			if not experiment.finished and experiment.task != 'test': # 'train' or 'both'
				experiment.run()
			if experiment.task != 'train': # 'test' or 'both'
				experiment.evaluate()
		# Clear session
		K.clear_session()

	def run_all(self, gpu_number=0):
		"""
		Execute all the experiments
		:param gpu_number: GPU that will be used.
		:return: None
		"""
		for experiment in self._generate_experiments():
			process_eval = multiprocessing.Process(target=self._run_one,args=(gpu_number,experiment,))
			process_eval.start()
			process_eval.join()
