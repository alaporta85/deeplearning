import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from matplotlib.image import imread
import matplotlib.pyplot as plt


def create_sets(layers):

	"""
	   Separate training and dev sets together with their labels, normalize
	   them and initialize parameters according to their shape.

	   Inputs:
	   - layers = list containing the size of each hidden layer (input and
				  output layers excluded). Ex: [16, 32].

	   Outputs:
	   - training_set = numpy array where each column represents an image. If
						there are m images with shape (nx, ny) than the shape
						of training_set is (nx * ny, m).

	   - training_lab = numpy array where each column represents the label of
						the corresponding image. Shape is (10, m). Each column
						is an one-hot vector. In this specific problem there
						are 10 rows because it is for 0-9 hand-written digits
						recognition.

	   - devel_set    = same as training_set. m can be different here.

	   - devel_lab    = same as training_lab. m can be different here.

	   - params       = dict of numpy arrays representing weights (W) and
						biases (b) for each layer.
	"""

	# Separate images and labels and shuffle them
	digits = load_digits()
	images = digits.images
	labels = digits.target
	images, labels = shuffle(images, labels, random_state=0)

	# im = imread('/Users/andrea/Desktop/prova5.png')
	# img = np.sum(im, axis=-1)
	# img = img.reshape(64, 1) / 255

	# Number of samples in training_set. Rest of the samples will be assigned
	# to devel_set
	m = 1000
	training_set = images[:m]  # Shape: (m, nx, ny)
	training_lab = from_value_to_onehot(labels[:m])  # Shape: (m, 10)
	devel_set = images[m:]  # Shape: (all - m, nx, ny)
	devel_lab = from_value_to_onehot(labels[m:])  # Shape: (all - m, 10)

	# Reshape and normalize sets
	training_set = training_set.reshape(
			training_set.shape[0], -1).T / 255  # Shape: (nx * ny, m)
	training_lab = training_lab.reshape(
			training_lab.shape[0], -1).T  # Shape: (10, m)
	devel_set = devel_set.reshape(
			devel_set.shape[0], -1).T / 255  # Shape: (nx * ny, all - m)
	devel_lab = devel_lab.reshape(
			devel_lab.shape[0], -1).T  # Shape: (10, all - m)

	# Add input and output layers' sizes to the input list "layers"
	layers = [training_set.shape[0]] + layers + [training_lab.shape[0]]

	# Initialize dict with parameters
	params = initialize_params_xavier(layers)

	return training_set, training_lab, devel_set, devel_lab, params


def from_value_to_onehot(some_array):

	"""
	   Return the one-hot version of the input array.

	   Inputs:
	   - some_array = numpy array of shape (m, 1).

	   Outputs:
	   - oh         = one-hot numpy array of shape (m, 10).

	   Ex:

			some_array = [1]           oh = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
						 [2]                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
						 [3]                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

	"""
	oh = np.zeros((len(some_array), 10))
	oh[np.arange(len(some_array)), some_array] = 1

	return oh


def from_array_to_onehot(some_array):

	"""
	   Return the one-hot version of the input array. The component which will
	   be equal to 1 is the one with the max value inside some_array. Used to
	   calculate the accuracy.

	   Inputs:
	   - some_array = numpy array of shape (10, m).

	   Outputs:
	   - oh         = one-hot numpy array of shape (10, m).

	   Ex:

	   some_array = [0.13, 0.98, 0.12, 0.05, 0.2, 0.17, 0.11, 0.22, 0.21, 0.15]
					[0.13, 0.12, 0.98, 0.05, 0.2, 0.17, 0.11, 0.22, 0.21, 0.15]
					[0.13, 0.05, 0.12, 0.98, 0.2, 0.17, 0.11, 0.22, 0.21, 0.15]

	   oh         = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	"""
	max_index = np.argmax(some_array)
	oh = np.zeros(some_array.shape)
	oh[max_index, :] = 1

	return oh


def initialize_params_xavier(layers):

	"""
	   Inputs:
	   - layers = list containing the size of each hidden layer (input and
				  output layers excluded). Ex: [16, 32].

	   Outputs:
	   - params = dict with W and b Xavier-initialized for each layer.
	"""

	params = {}
	np.random.seed(1)

	for i in range(1, len(layers)):
		xavier = np.sqrt(1 / layers[i - 1])
		params['W' + str(i)] = np.random.randn(layers[i - 1],
		                                       layers[i]) * xavier
		params['b' + str(i)] = np.zeros((layers[i], 1))

	return params


def softmax(z):

	"""
	   Inputs:
	   - z  = numpy array

	   Outputs:
	   - s  = numpy array
	   - z
	"""

	s = np.divide(np.exp(z), np.sum(np.exp(z), axis=0))

	return s, z


def relu(z):

	"""
	   Inputs:
	   - z  = numpy array

	   Outputs:
	   - r  = numpy array
	   - z
	"""

	r = np.maximum(0, z)

	return r, z


def softmax_back(a, labels):

	"""
	   Only in the case it is applied to the NN output layer, formula is
	   a - labels.

	   Inputs:
	   - a      = numpy array. Output of the NN output layer.

	   - labels = numpy array.

	   Outputs:
	   - dz = numpy array.
	"""

	dz = np.subtract(a, labels)

	return dz


def relu_back(da, act_cache):

	"""
	   Inputs:
	   - da        = numpy array.

	   - act_cache = numpy array. It is current layer's Z.

	   Outputs:
	   - dz = numpy array.
	"""

	dz = da * (act_cache >= 0)

	return dz


def forward_lin(a_previous, w, b):

	"""
	   Compute the forward linear step relative to the current layer (l).

	   Inputs:
	   - a_previous = numpy array. Output of NN layer [l - 1].

	   - w          = numpy array. Weights of NN layer [l].

	   - b          = numpy array. Biases of NN layer [l].

	   Outputs:
	   - z         = numpy array

	   - cache_lin = tuple
	"""

	z = np.dot(w.T, a_previous) + b
	cache_lin = (a_previous, w, b)

	return z, cache_lin


def forward_act(a_previous, w, b, activation):

	"""
	   Compute the forward activation step relative to the current layer (l).

	   Inputs:
	   - a_previous = numpy array. Output of NN layer [l - 1].

	   - w          = numpy array. Weights of NN layer [l].

	   - b          = numpy array. Biases of NN layer [l].

	   - activation = string. 'relu' or 'softmax'

	   Outputs:
	   - a         = numpy array. Output of NN layer [l].

	   - _cache = tuple
	"""

	if activation == 'relu':
		z, cache_lin = forward_lin(a_previous, w, b)
		a, cache_act = relu(z)
	else:
		z, cache_lin = forward_lin(a_previous, w, b)
		a, cache_act = softmax(z)

	_cache = (cache_lin, cache_act)

	return a, _cache


def forward_prop(a_previous, params):

	"""
	   Compute the forward propagation through the entire NN.

	   Inputs:
	   - a_previous = numpy array. Output of NN layer [l - 1].

	   - params     = dict with W and b.

	   Outputs:
	   - a_final = numpy array. Output of NN's output layer (the prediction).

	   - caches  = list of the caches stored as tuples. One tuple per NN layer.
	"""

	# Number of hidden layers
	h = len(params) // 2
	caches = []

	# Compute the forward prop for all the layers except the last one. At this
	# point we use just relu
	for i in range(1, h):
		a_previous, _cache = forward_act(a_previous, params['W' + str(i)],
		                                 params['b' + str(i)], 'relu')
		caches.append(_cache)

	# Compute the forward prop for last layer by using softmax
	a_final, _cache = forward_act(a_previous, params['W' + str(h)],
	                              params['b' + str(h)], 'softmax')
	caches.append(_cache)

	return a_final, caches


def compute_cost(a_final, labels):

	"""
	   Compute the cost function.

	   Inputs:
	   - a_final = numpy array. Prediction after forward prop through NN.

	   - labels  = numpy array.

	   Outputs:
	   - cost = float.
	"""

	cost = -np.sum(labels * np.log(a_final)) / labels.shape[1]

	return cost


def backward_lin(dz, cache_lin):

	"""
	   Compute the backward linear step relative to the current layer (l).

	   Inputs:
	   - dz        = numpy array.

	   - cache_lin = tuple

	   Outputs:
	   - z         = numpy array

	   - cache_lin = tuple
	"""
	a_previous, w, b = cache_lin
	m = a_previous.shape[1]

	dw = (np.dot(dz, a_previous.T)) / m
	db = np.sum(dz, axis=1, keepdims=True) / m
	da_previous = np.dot(w, dz)

	return da_previous, dw, db


def backward_act(da_previous, current_cache, labels, activation):

	"""
	   Compute the backward activation step relative to the current layer (l).

	   Inputs:
	   - da_previous   = numpy array. Relative to layer [l].

	   - current_cache = tuple.

	   - labels        = numpy array.

	   - activation = string. 'relu' or 'softmax'

	   Outputs:
	   - da_previous   = numpy array. Relative to layer [l - 1].

	   - dw            = numpy array. Weights gradients of NN layer [l].

	   - db            = numpy array. Biases gradients of NN layer [l].
	"""

	cache_lin, cache_act = current_cache
	a, w, b = cache_lin
	z = cache_act

	if activation == 'relu':
		dz = relu_back(da_previous, z)
		da_previous, dw, db = backward_lin(dz, cache_lin)
	else:
		dz = softmax_back(a, labels)
		da_previous, dw, db = backward_lin(dz, cache_lin)

	return da_previous, dw, db


def back_prop(a_final, labels, caches):

	"""
	   Compute the backward propagation through the entire NN.

	   Inputs:
	   - a_final = numpy array. Output of forward propagation.

	   - labels  = numpy array.

	   - caches = list of tuples.

	   Outputs:
	   - grads = dict containing the gradients of W and b.
	"""

	grads = {}
	L = len(caches)

	# First we load the last cache and delete it
	last_lin_cache, last_act_cache = caches[-1]
	_cache = caches[:-1]

	# Then only for this layer apply softmax_back
	dz_prev = softmax_back(a_final, labels)
	da_previous, dw, db = backward_lin(dz_prev, last_lin_cache)
	grads['dW{}'.format(L)] = dw
	grads['db{}'.format(L)] = db
	L = len(_cache)

	# For the rest of the layers apply relu_back
	for i in reversed(range(L)):
		current_lin_cache, current_act_cache = _cache[i]
		dz_prev = relu_back(da_previous, current_act_cache)
		da_previous, dw, db = backward_lin(dz_prev, current_lin_cache)
		grads['dW{}'.format(i + 1)] = dw
		grads['db{}'.format(i + 1)] = db

	return grads


def update_parameters(params, grads, learning_rate):

	"""
	   Update W and b after back propagation.

	   Inputs:
	   - params        = dict containing W and b for each NN's layer.

	   - grads         = dict containing dW and db for each NN's layer.

	   - learning_rate = float. Typical value is 0.01

	   No Outputs.
	"""
	L = len(params) // 2

	for i in range(1, L + 1):
		params['W{}'.format(i)] -= learning_rate * grads['dW{}'.format(i)].T
		params['b{}'.format(i)] -= learning_rate * grads['db{}'.format(i)]


def forw_back_upd(a0, labels, params, learning_rate):

	"""
	   Compute forward prop, cost, back prop and update W and b.

	   Inputs:
	   - a0            = numpy array. The input set of images.

	   - labels        = numpy array.

	   - params        = dict containing W and b.

	   - learning_rate = float.

	   Outputs:
	   - cost = float.
	"""

	a_final, cache = forward_prop(a0, params)
	cost = compute_cost(a_final, labels)
	grads = back_prop(a_final, labels, cache)
	update_parameters(params, grads, learning_rate)

	return cost


def compute_accuracy(a_final, labels, which_set):

	"""
	   Compute accuracy of the model.

	   Inputs:
	   - a_final = numpy array.

	   - labels  = numpy array.

	   Outputs:
	   - accuracy = float.
	"""

	nx = a_final.shape[0]
	m = a_final.shape[1]
	count = 0

	for x in range(m):
		prevision = from_array_to_onehot(a_final[:, x].reshape(nx, 1))
		label = labels[:, x].reshape(nx, 1)

		if np.array_equal(prevision, label):
			count += 1

	accuracy = round(count / m * 100, 1)

	print('Accuracy for {} set: {} %'.format(which_set, accuracy))


def model_plain(input_set, input_lab, params, learning_rate,
                iterations, which_set):

	all_costs = []

	for i in range(iterations + 1):
		cost = forw_back_upd(input_set, input_lab, params, learning_rate)
		all_costs.append(cost)

		if i % 5000 == 0:
			print("Cost at iteration {}: {}".format(i, round(cost, 4)))

	a_final, _ = forward_prop(input_set, params)
	compute_accuracy(a_final, input_lab, which_set)


def model_mini_batch(input_set, input_lab, params, learning_rate,
                     epochs, batch_size, which_set):

	all_cost = []
	mini_batches = input_set.shape[1] // batch_size + 1 if \
		input_set.shape[1] % batch_size else input_set.shape[1] // batch_size

	for i in range(epochs + 1):
		cost = 0
		for x in range(mini_batches):
			inp = input_set[:, x * batch_size: (x + 1) * batch_size]
			label = input_lab[:, x * batch_size: (x + 1) * batch_size]
			cost += forw_back_upd(inp, label, params, learning_rate)
		all_cost.append(cost / mini_batches)

		if i % 5000 == 0:
			print("Cost at epoch {}: {}".format(i, round(cost, 4)))

	a_final, _ = forward_prop(input_set, params)
	compute_accuracy(a_final, input_lab, which_set)


def predict_image(image, params):
	img = np.sum(image, axis=-1)
	img = img.reshape(64, 1) / 255

	a_final, _ = forward_prop(img, params)
	res = from_array_to_onehot(a_final)

	print('Picture is a {}!'.format(np.argmax(res)))


hidden_layers = [64, 64, 32]

# train_set, train_lab, dev_set, dev_lab, parameters = create_sets(hidden_layers)
# model_plain(train_set, train_lab, parameters, 0.01, 15000, 'train')
# model_plain(dev_set, dev_lab, parameters, 0.01, 1, 'dev')


train_set, train_lab, dev_set, dev_lab, parameters = create_sets(hidden_layers)
model_mini_batch(train_set, train_lab, parameters, 0.01, 20000, 512, 'train')
# model_mini_batch(dev_set, dev_lab, parameters, 0.01, 1, 512, 'dev')

# im = imread('/Users/andrea/Desktop/prova4.png')
# predict_image(im, parameters)
