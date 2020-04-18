import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Lambda,Input
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from activations import SPP, SPPT, MPELU, RTReLU, RTPReLU, PairedReLU, EReLU, SQRTActivation, CLM, RReLu, PELU, SlopedReLU, PTELU, Antirectifier, CReLU, EPReLU
from layers import GeometricLayer, ScaleLayer
from resnet import Resnet_2x4

from inception_resnet_v2 import InceptionResNetV2 as Irnv2
from unimodal_extensions import _add_binom_m


class Net:
	def __init__(self, size, activation, final_activation, f_a_params={}, use_tau=True, prob_layer=None, num_channels=3,
				 num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.final_activation = final_activation
		self.f_a_params = f_a_params
		self.use_tau = use_tau
		self.prob_layer = prob_layer
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

	def build(self, net_model):
		if hasattr(self, net_model):
			return getattr(self, net_model)()
		else:
			raise Exception('Invalid network model.')

	def vgg19(self):

		input = tf.keras.layers.Input(shape=(self.size, self.size, self.num_channels))
		# Block 1
		x=Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
							input_shape=(self.size, self.size, self.num_channels), data_format='channels_last')(input),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=MaxPooling2D()(x),

		# Block 2
		x=Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=MaxPooling2D()(x),

		# Block 3
		x=Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=MaxPooling2D()(x),

		# Block 4
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=MaxPooling2D()(x),

		# Block 5
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', )(x),
		x=self.__activation()(x),
		x=BatchNormalization()(x),
		x=MaxPooling2D()(x),

		# Classification block
		x=Flatten()(x),
		x=Dropout(rate=self.dropout)(x),
		x=Dense(4096)(x),
		x=self.__activation()(x),
		x=Dense(4096)(x),
		x=self.__activation()(x),
		

		x = self.__final_activation(x)
		model = tf.keras.Model(input, x)		
		return model

	def vgg16(self):
		weight_decay = 0.0005
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = tf.keras.Sequential([


			Conv2D(64, (3, 3), padding='same',
							 input_shape=(self.size, self.size, self.num_channels), kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.3),

			Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			MaxPooling2D(pool_size=(2, 2)),

			Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			MaxPooling2D(pool_size=(2, 2)),

			Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			MaxPooling2D(pool_size=(2, 2)),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			MaxPooling2D(pool_size=(2, 2)),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),
			Dropout(0.4),

			Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			MaxPooling2D(pool_size=(2, 2)),
			Dropout(0.5),

			Flatten(),
			Dense(512, kernel_regularizer=regularizers.l2(weight_decay)),
			self.__activation(),
			BatchNormalization(),

			Dropout(0.5),
			Dense(self.num_classes),
			Activation('softmax'),
		])



		return model

	def vgg16_32f(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=(self.size, self.size, self.num_channels), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(Dense(32, kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		return model

	def vgg16pu(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=(self.size, self.size, self.num_channels), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.3))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		model.add(Flatten())
		model.add(ScaleLayer())
		model.add(Lambda(lambda x: K.log(x)))
		model.add(Dense(16))
		model.add(Lambda(lambda x: K.exp(x)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(Dropout(0.5))
		model.add(Dense(self.num_classes))
		model.add(Activation('softmax'))

		return model


	def conv128(self):

		feature_filter_size = 3
		classif_filter_size = 4

		input = Input(shape=(self.size, self.size, self.num_channels))

		x = Conv2D(32, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(input)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = Conv2D(32, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D()(x)

		x = Conv2D(64, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = Conv2D(64, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D()(x)

		x = Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D()(x)

		x = Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = Conv2D(128, feature_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D()(x)

		x = Conv2D(128, classif_filter_size, strides=(1, 1), kernel_initializer='he_uniform')(x)
		x = self.__activation()(x)
		x = BatchNormalization()(x)

		x = Flatten()(x)

		x = Dense(96)(x)

		if self.dropout > 0:
			x = Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = tf.keras.Model(input, x)

		return model

	def inceptionresnetv2(self):
		input = Input(shape=(self.size, self.size, self.num_channels))
		x = input
		# Required size >= 75 x 75
		size = self.size
		if size < 75:
			size = 75
			x = tf.keras.layers.ZeroPadding2D(padding=(75 - self.size) // 2 + 1)(x)

		x = Irnv2(input_tensor=x, include_top=False, input_shape=(size, size, self.num_channels),
				  classes=self.num_classes, pooling='avg', activation=self.__activation())(x)

		x = tf.keras.layers.Dense(512)(x)

		if self.dropout > 0:
			x = tf.keras.layers.Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = tf.keras.Model(input, x)

		return model

	def beckhamresnet(self):
		input = tf.keras.layers.Input(shape=(self.size, self.size, self.num_channels))
		x = input

		resnet = Resnet_2x4((self.size, self.size, self.num_channels), activation=self.__activation())
		x = resnet.get_net()(x)

		if self.dropout > 0:
			x = tf.keras.layers.Dropout(rate=self.dropout)(x)

		x = self.__final_activation(x)

		model = tf.keras.Model(input, x)

		return model

	def __activation(self):
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
			return SPP(self.spp_alpha)
		elif self.activation == 'sppt':
			return SPPT()
		elif self.activation == 'mpelu':
			return MPELU(channel_wise=True)
		elif self.activation == 'rtrelu':
			return RTReLU()
		elif self.activation == 'rtprelu':
			return RTPReLU()
		elif self.activation == 'pairedrelu':
			return PairedReLU()
		elif self.activation == 'erelu':
			return EReLU()
		elif self.activation == 'eprelu':
			return EPReLU()
		elif self.activation == 'sqrt':
			return SQRTActivation()
		elif self.activation == 'rrelu':
			return RReLu()
		elif self.activation == 'pelu':
			return PELU()
		elif self.activation == 'slopedrelu':
			return SlopedReLU()
		elif self.activation == 'ptelu':
			return PTELU()
		elif self.activation == 'antirectifier':
			return Antirectifier()
		elif self.activation == 'crelu':
			return CReLU()
		else:
			return tf.keras.layers.Activation('relu')

	def __final_activation(self, x):
		if self.final_activation == 'poml':
			x = Dense(1)(x)
			x = BatchNormalization()(x)
			x = CLM(self.num_classes, 'logit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomp':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'probit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomclog':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'cloglog', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'pomglogit':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'glogit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmcauchit':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'cauchit', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmggamma':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'ggamma', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmgauss':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'gauss', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'clmexpgauss':
			x = tf.keras.layers.Dense(1)(x)
			x = tf.keras.layers.BatchNormalization()(x)
			x = CLM(self.num_classes, 'expgauss', self.f_a_params, use_tau=self.use_tau)(x)
		elif self.final_activation == 'binomial':
			_add_binom_m(model, self.num_classes, 1.0, 'sigm_learnable')
		else:
			x = Dense(self.num_classes)(x)
			if self.prob_layer == 'geometric':
				x = GeometricLayer()(x)
			x = tf.keras.layers.Activation(self.final_activation)(x)

		return x
