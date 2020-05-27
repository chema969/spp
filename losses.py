import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

@tf.function
def make_cost_matrix(num_ratings):
	"""
	Create a quadratic cost matrix of num_ratings x num_ratings elements.

	:param thresholds_b: threshold b1.
	:param thresholds_a: thresholds alphas vector.
	:param num_labels: number of labels.
	:return: cost matrix.
	"""

	cost_matrix = np.reshape(np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings))
	cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_ratings - 1) ** 2.0
	return np.float32(cost_matrix)


def qwk_loss(cost_matrix):
	"""
	Compute QWK loss function.

	:param pred_prob: predict probabilities tensor.
	:param true_prob: true probabilities tensor.
	:param cost_matrix: cost matrix.
	:return: QWK loss value.
	"""
	@tf.function
	def _qwk_loss(true_prob, pred_prob):
		targets = K.argmax(true_prob, axis=1)
		costs = K.gather(cost_matrix, targets)

		numerator = costs * pred_prob
		numerator = K.sum(numerator)

		sum_prob = K.sum(pred_prob, axis=0)
		n = K.sum(true_prob, axis=0)

		a = K.reshape(K.dot(cost_matrix, K.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
		b = K.reshape(n / K.sum(n), shape=[-1])

		epsilon = 10e-9

		denominator = a * b
		denominator = K.sum(denominator) + epsilon

		return numerator / denominator # + K.cast(K.sum(conf_mat) * 0, dtype=K.floatx())

	return _qwk_loss


@tf.function
def _compute_sensitivities(y_true, y_pred):
	"""
	Compute the weighted sensitivities 

	:param y_true: true class.
	:param y_pred: predicted class.
	:return: Weighted sensitivities value.
	"""

	diff = (1.0 - K.pow(y_true - y_pred, 2)) / 2.0 # [0,1]
	diff_class = K.sum(diff, axis=1) # vector of size N
	sum = K.sum(diff_class) # total sum of that vector
	sensitivities = diff_class / sum

	return sensitivities

@tf.function
def ms_loss(true_prob, pred_prob):
	"""
	Compute mean sensitivities loss function.

	:param pred_prob: predict probabilities tensor.
	:param true_prob: true probabilities tensor.
	:return: ms loss value.
	"""    
	sensis = _compute_sensitivities(true_prob, pred_prob)
	mean_sens = K.mean(sensis)
	return mean_sens


def ms_n_qwk_loss(qwk_cost_matrix, alpha=0.5):
	"""
	Compute the weighted sum of mean sensitivities and quadratic weighted kappa loss.

	:param qwk_cost_matrix: cost matrix.
	:param alpha: weight for qwk in comparaison with ms. It must be between 1 and 0.
	:return: the weighted sum of mean sensitivities and quadratic weighted kappa loss.
	"""    
	@tf.function
	def _ms_n_qwk_loss(true_prob, pred_prob):
		qwk = qwk_loss(qwk_cost_matrix)(true_prob, pred_prob)
		ms = ms_loss(true_prob, pred_prob)

		return alpha * qwk + (1 - alpha) * ms

	return _ms_n_qwk_loss
