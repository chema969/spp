import tensorflow as tf


def parametric_softplus(spp_alpha):
	def spp(x):
		return tf.log(1 + tf.exp(x)) - spp_alpha
	return spp

class Net:
	def __init__(self, size, activation, num_channels=3, num_classes=5, spp_alpha=0.2):
		self.size = size
		self.activation = activation
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha

		# Add new activation function
		tf.keras.utils.get_custom_objects().update({'spp': tf.keras.layers.Activation(parametric_softplus(spp_alpha))})

	def vgg19(self):
		model = tf.keras.Sequential([
			# Block 1
			tf.keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
								input_shape=(self.size, self.size, self.num_channels), data_format='channels_last'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 2
			tf.keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 3
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 4
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 5
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Classification block
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(4096),
			self.__get_activation(),
			tf.keras.layers.Dense(4096),
			self.__get_activation(),
			tf.keras.layers.Dense(4096, activation='softmax'),

		])

		return model
		
	def __get_activation(self):
		if self.activation == 'relu':
			return tf.keras.layers.Activation('relu')
		elif self.activation == 'lrelu':
			return tf.keras.layers.LeakyReLU()
		elif self.activation == 'prelu':
			return tf.keras.layers.PReLU()
		elif self.activation == 'elu':
			return tf.keras.layers.ELU()
		elif self.activation == 'softplus':
			return tf.keras.layers.Activation('softplus')
		elif self.activation == 'spp':
			return tf.keras.layers.Activation('spp')
		else:
			return tf.keras.layers.Activation('relu')
