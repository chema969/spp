import click
import datetime
from experiment import Experiment
from experimentset import ExperimentSet
import tensorflow as tf

@click.group()
def cli():
	pass


@cli.command('experiment', help='Experiment mode')
@click.option('--file', '-f', required=True, help=u'File that contains the experiments that will be executed.')
@click.option('--gpu', '-g', required=False, default=0, help=u'GPU index')
def experiment(file, gpu):
	begin=datetime.datetime.now()
	#Allow soft device placement, so in case the device is not found, TensorFlow will automatically choose an existing and supported device
	tf.config.set_soft_device_placement(True)
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	try: 
		tf.config.experimental.set_memory_growth(physical_devices[gpu], True)
	except: 
 		# Invalid device or cannot modify virtual devices once initialized. 
  		pass 

	experimentSet = ExperimentSet(file)
	experimentSet.run_all(gpu_number=gpu)
	print(datetime.datetime.now()-begin)

if __name__ == '__main__':
	cli()
