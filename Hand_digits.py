import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def create_sets(m):

	"""
	   Separate training and dev sets together with their labels, normalize
	   them and initialize parameters according to their shape.

	   Inputs:
	   - m = number of samples used for training.

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
	"""

	x, y = loadlocal_mnist(
		images_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_train_set',
		labels_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_train_labels')

	w, z = loadlocal_mnist(
		images_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_dev_set',
		labels_path='/Users/andrea/Desktop/deeplearning.ai/MNIST_dev_labels')

	training_set = x[:m, :].T / 255
	training_lab = from_value_to_onehot(y[:m])
	training_lab = training_lab.T

	devel_set = w[:m, :].T / 255
	devel_lab = from_value_to_onehot(z[:m])
	devel_lab = devel_lab.T

	return training_set, training_lab, devel_set, devel_lab


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


def initialize_params(training_set, training_lab, layers,
                      initialization='xavier', seed=True):

	"""
	   Inputs:
	   - training_set   = numpy array of shape (nx * ny, m).

	   - training_lab   = numpy array of shape (n_output, m).

	   - layers         = list containing the size of each hidden layer (input
	                      and output layers excluded). Ex: [16, 32].

	   - initialization = float or string.

	   - seed           = bool. If True, seed is used in random initialization.

	   Outputs:
	   - params = dict of numpy arrays representing weights (W) and biases (b)
	              for each layer.
	"""

	if seed:
		np.random.seed(1)

	params = {}

	# Add input and output layers' sizes to the input list "layers"
	layers = [training_set.shape[0]] + layers + [training_lab.shape[0]]

	for i in range(1, len(layers)):
		if initialization == 'xavier':
			factor = np.sqrt(1 / layers[i - 1])
		else:
			factor = initialization

		params['W' + str(i)] = np.random.randn(layers[i - 1],
		                                       layers[i]) * factor
		params['b' + str(i)] = np.zeros((layers[i], 1))

	return params


def initialize_velocities(grads):

	"""
	   Initialize with zeros the velocities for RMSprop and Adam optimizers.

	   Inputs:
	   - grads = dict containing dW and db for each NN's layer.

	   Outputs:
	   - velocities = dict containing dW and db for each NN's layer.
	"""

	L = len(grads) // 2
	velocities = {}

	for i in range(1, L + 1):
		dw = grads['dW{}'.format(i)]
		db = grads['db{}'.format(i)]
		velocities['dW{}'.format(i)] = np.zeros(dw.shape)
		velocities['db{}'.format(i)] = np.zeros(db.shape)

	return velocities


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


def update_parameters(params, grads, learning_rate, iteration,
                      beta1, beta2, epsilon, optimizer):

	"""
	   Update W and b after back propagation.

	   Inputs:
	   - params        = dict containing W and b for each NN's layer.

	   - grads         = dict containing dW and db for each NN's layer.

	   - learning_rate = float. Typical value is 0.01

	   - iteration     = int. Running iteration.

	   - beta1         = float. Used in GDM and Adam. Typical value is 0.9

	   - beta2         = float. Used in RMS and Adam. Typical value is 0.999

	   - epsilon       = float. Used in RMS and Adam. Typical value is 1e-8

	   - optimizer     = string. "GD", "GDM", "RMS" or "ADAM".

	   No Outputs.
	"""

	L = len(params) // 2

	# Calculate bias correction
	bias_corr1 = (1 - beta1 ** iteration)
	bias_corr2 = (1 - beta2 ** iteration)

	# Set all gradients values depending on the optimizer
	if optimizer == 'GD':
		pass

	elif optimizer == 'GDM':
		v = initialize_velocities(grads)
		for i in range(1, L + 1):
			vdw = v['dW{}'.format(i)]
			vdb = v['db{}'.format(i)]

			v['dW{}'.format(i)] = (beta1 * vdw + (1 - beta1) *
			                       grads['dW{}'.format(i)])

			v['db{}'.format(i)] = (beta1 * vdb + (1 - beta1) *
			                       grads['db{}'.format(i)])

			grads['dW{}'.format(i)] = v['dW{}'.format(i)]
			grads['db{}'.format(i)] = v['db{}'.format(i)]

	elif optimizer == 'RMS':
		s = initialize_velocities(grads)
		for i in range(1, L + 1):
			sdw = s['dW{}'.format(i)]
			sdb = s['db{}'.format(i)]

			s['dW{}'.format(i)] = (beta2 * sdw + (1 - beta2) *
			                       np.square(grads['dW{}'.format(i)]))

			s['db{}'.format(i)] = (beta2 * sdb + (1 - beta2) *
			                       np.square(grads['db{}'.format(i)]))

			grads['dW{}'.format(i)] /= (np.sqrt(s['dW{}'.format(i)]) + epsilon)
			grads['db{}'.format(i)] /= (np.sqrt(s['db{}'.format(i)]) + epsilon)

	elif optimizer == 'ADAM':
		v = initialize_velocities(grads)
		s = initialize_velocities(grads)
		for i in range(1, L + 1):
			vdw = v['dW{}'.format(i)]
			vdb = v['db{}'.format(i)]
			sdw = s['dW{}'.format(i)]
			sdb = s['db{}'.format(i)]

			v['dW{}'.format(i)] = (beta1 * vdw + (1 - beta1) *
			                       grads['dW{}'.format(i)])
			v['db{}'.format(i)] = (beta1 * vdb + (1 - beta1) *
			                       grads['db{}'.format(i)])
			s['dW{}'.format(i)] = (beta2 * sdw + (1 - beta2) *
			                       np.square(grads['dW{}'.format(i)]))
			s['db{}'.format(i)] = (beta2 * sdb + (1 - beta2) *
			                       np.square(grads['db{}'.format(i)]))

			vdw_corr = v['dW{}'.format(i)] / bias_corr1
			vdb_corr = v['db{}'.format(i)] / bias_corr1
			sdw_corr = s['dW{}'.format(i)] / bias_corr2
			sdb_corr = s['db{}'.format(i)] / bias_corr2

			grads['dW{}'.format(i)] = vdw_corr / (np.sqrt(sdw_corr) + epsilon)
			grads['db{}'.format(i)] = vdb_corr / (np.sqrt(sdb_corr) + epsilon)

	# Update W and b
	for i in range(1, L + 1):
		params['W{}'.format(i)] -= learning_rate * grads['dW{}'.format(i)].T
		params['b{}'.format(i)] -= learning_rate * grads['db{}'.format(i)]


def forw_back_upd(a0, labels, params, learning_rate,
                  iteration, beta1, beta2, epsilon, optimizer):

	"""
	   Compute forward prop, cost, back prop and update W and b.

	   Inputs:
	   - a0            = numpy array. The input set of images.

	   - labels        = numpy array.

	   - params        = dict containing W and b.

	   - learning_rate = float.

	   - iteration     = int. Running iteration.

	   - beta1         = float. Used in GDM and Adam. Typical value is 0.9

	   - beta2         = float. Used in RMS and Adam. Typical value is 0.999

	   - epsilon       = float. Used in RMS and Adam. Typical value is 1e-8

	   - optimizer     = string. "GD", "GDM", "RMS" or "ADAM".

	   Outputs:
	   - cost = float.
	"""

	a_final, cache = forward_prop(a0, params)
	cost = compute_cost(a_final, labels)
	grads = back_prop(a_final, labels, cache)
	update_parameters(params, grads, learning_rate,
	                  iteration, beta1, beta2, epsilon, optimizer)

	return cost


def compute_accuracy(a_final, labels, which_set, print_accuracy):

	"""
	   Compute accuracy of the model.

	   Inputs:
	   - a_final        = numpy array.

	   - labels         = numpy array.

	   - which_set      = string.

	   - print_accuracy = bool.

	   No Outputs.
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

	if print_accuracy:
		print('Accuracy for {} set: {} %'.format(which_set, accuracy))


def model_plain(input_set, input_lab, params, learning_rate,
                learning_rate_decay, iterations, beta1, beta2, epsilon,
                optimizer, print_every, plot_cost=False, print_accuracy=False):
	"""
	   Train the model.

	   Inputs:
	   - input_set           = numpy array. The input set of images.

	   - input_lab           = numpy array.

	   - params              = dict containing W and b.

	   - learning_rate       = float.

	   - learning_rate_decay = float.

	   - iterations          = int.

	   - beta1               = float. Used in GDM/Adam. Typical value is 0.9

	   - beta2               = float. Used in RMS/Adam. Typical value is 0.999

	   - epsilon             = float. Used in RMS/Adam. Typical value is 1e-8

	   - optimizer           = string. "GD", "GDM", "RMS" or "ADAM".

	   - print_every         = int. Number of iterations to print the cost.
	                           If == "last", only last cost will be printed.

	   - plot_cost           = bool.

	   - print_accuracy      = bool.

	   Outputs:
	   - all_cost = list.
	"""

	all_costs = []

	for i in range(iterations + 1):
		alpha = (1 / (1 + learning_rate_decay * i)) * learning_rate
		cost = forw_back_upd(input_set, input_lab, params, alpha,
		                     i + 1, beta1, beta2, epsilon, optimizer)
		all_costs.append(cost)

		if type(print_every) == int and i % print_every == 0:
			print("Cost at iteration {}: {}".format(i, round(cost, 4)))

	if print_every == 'last':
		print("Cost at last epoch for {} optimizer: {}".format(
				optimizer, round(cost, 4)))

	a_final, _ = forward_prop(input_set, params)
	compute_accuracy(a_final, input_lab, 'training', print_accuracy)

	if plot_cost:
		plt.plot(all_costs, label='Plain, {}'.format(optimizer))
		plt.legend()
		plt.show()

	return all_costs


def model_mini_batch(input_set, input_lab, params, learning_rate,
                     learning_rate_decay, epochs, batch_size,
                     beta1, beta2, epsilon, optimizer, print_every,
                     plot_cost=False, print_accuracy=False):
	"""
	   Train the model.

	   Inputs:
	   - input_set           = numpy array. The input set of images.

	   - input_lab           = numpy array.

	   - params              = dict containing W and b.

	   - learning_rate       = float.

	   - learning_rate_decay = float.

	   - epochs              = int.

	   - batch_size          = int.

	   - beta1               = float. Used in GDM/Adam. Typical value is 0.9

	   - beta2               = float. Used in RMS/Adam. Typical value is 0.999

	   - epsilon             = float. Used in RMS/Adam. Typical value is 1e-8

	   - optimizer           = string. "GD", "GDM", "RMS" or "ADAM".

	   - print_every         = int. Number of iterations to print the cost.
	                           If == "last", only last cost will be printed.

	   - plot_cost           = bool.

	   - print_accuracy      = bool.

	   Outputs:
	   - all_cost = list.
	"""

	cost_epochs = []
	cost_batches = []
	mini_batches = input_set.shape[1] // batch_size + 1 if \
		input_set.shape[1] % batch_size else input_set.shape[1] // batch_size

	for i in range(epochs + 1):
		alpha = (1 / (1 + learning_rate_decay * i)) * learning_rate
		cost = 0
		for x in range(mini_batches):
			inp = input_set[:, x * batch_size: (x + 1) * batch_size]
			label = input_lab[:, x * batch_size: (x + 1) * batch_size]
			cost += forw_back_upd(inp, label, params, alpha,
			                      i + 1, beta1, beta2, epsilon, optimizer)
			cost_batches.append(cost)
		cost /= mini_batches
		cost_epochs.append(cost)

		if type(print_every) == int and i % print_every == 0:
			print("Cost at epoch {}: {}".format(i, round(cost, 4)))

	if print_every == 'last':
		print("Cost at last epoch for {} optimizer: {}".format(
				optimizer, round(cost, 4)))

	a_final, _ = forward_prop(input_set, params)
	compute_accuracy(a_final, input_lab, 'training', print_accuracy)

	if plot_cost == 'each_epoch':
		plt.plot(cost_epochs, label='MB, {}'.format(optimizer))
		plt.legend()
		plt.show()
	elif plot_cost == 'each_mini_batch':
		plt.plot(cost_batches, label='MB, {}'.format(optimizer))
		plt.legend()
		plt.show()

	return cost_batches, cost_epochs


def test_dev_set(devel_set, devel_lab, params):

	"""
	   Test the model on dev set.

	   Inputs:
	   - devel_set = numpy array.

	   - devel_lab = numpy array.

	   - params    = dict. Trained parameters.

	   No Outputs.
	"""

	a_final, _ = forward_prop(devel_set, params)
	compute_accuracy(a_final, devel_lab, 'dev', True)


# hidden_layers = [32, 32, 32]
# train_set, train_lab, dev_set, dev_lab = create_sets(m=60000)
# parameters = initialize_params(train_set, train_lab, hidden_layers,
#                                initialization='xavier', seed=True)

# model_plain(train_set, train_lab, parameters, learning_rate=0.01,
#             iterations=2, plot_cost=False)


# cost_batches_GD, cost_epochs_GD = model_mini_batch(
# 		train_set, train_lab, parameters, learning_rate=0.00001,
# 		learning_rate_decay=0.005, epochs=1, batch_size=1024,
# 		beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer='RMS',
# 		print_every='last', plot_cost=False, print_accuracy=False)
# test_dev_set(dev_set, dev_lab, parameters)
