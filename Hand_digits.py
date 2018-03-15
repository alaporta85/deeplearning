import numpy as np
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def create_sets(layers):

    digits = load_digits()

    images = digits.images
    labels = digits.target

    images, labels = shuffle(images, labels, random_state=0)
    max_labels = np.amax(labels)

    m = 1000

    training_set = images[:m]
    training_lab = one_hot(labels[:m], max_labels)
    devel_set = images[m:]
    devel_lab = one_hot(labels[m:], max_labels)

    training_set = training_set.reshape(training_set.shape[0], -1).T / 255
    training_lab = training_lab.reshape(training_lab.shape[0], -1).T
    devel_set = devel_set.reshape(devel_set.shape[0], -1).T / 255
    devel_lab = devel_lab.reshape(devel_lab.shape[0], -1).T

    # mu = np.average(training_set)
    # sigma = np.sum(np.square(training_set), axis=1, keepdims=True) / m
    # training_set -= mu
    # training_set /= sigma

    layers = [training_set.shape[0]] + layers + [training_lab.shape[0]]

    param = initialize_params(layers)

    return training_set, training_lab, devel_set, devel_lab, param


def create_practice_sets():

    a0 = np.array([[2, 5, 4, 3, 2, 1]]).T

    y = np.array([0, 1, 0]).reshape(3, 1)

    params = {'W1': np.array([[-.2, .3, .1],
                              [.1, -.1, .2],
                              [-.3, .1, -.4],
                              [.2, -.5, .4],
                              [.1, .4, -.5],
                              [-.1, .1, .5]]).reshape(3, 6),
              'W2': np.array([[-.2, .4, -.5],
                              [.3, -.2, -.5],
                              [.4, -.1, .5]]),
              'b1': np.zeros((3, 1)),
              'b2': np.zeros((3, 1))}

    return a0, y, params


def one_hot(some_array, size):
    oh = np.zeros((len(some_array), size + 1))
    oh[np.arange(len(some_array)), some_array] = 1
    return oh


def one_hot2(some_array):
    max_index = np.argmax(some_array)
    oh = np.zeros(some_array.shape)
    oh[max_index, :] = 1

    return oh


def initialize_params(layers):

    np.random.seed(25)
    factor1 = 0.01

    param = {}

    for i in range(1, len(layers)):
        factor2 = np.sqrt(2/layers[i - 1])
        param['W' + str(i)] = np.random.randn(layers[i],
                                              layers[i - 1]) * factor2
        param['b' + str(i)] = np.zeros((layers[i], 1))

    return param


def softmax(z):
    numer = np.exp(z)
    # denom = np.sum(numer)
    denom = np.sum(numer, axis=0)

    s = np.divide(numer, denom)
    return s, z


def relu(z):
    r = np.maximum(0, z)
    return r, z


def softmax_back(a, labels):

    return np.subtract(a, labels)


def relu_back(da, act_cache):

    dz = da * (act_cache >= 0)

    return dz


def forward_lin(a, w, b):
    z = np.dot(w, a) + b
    cache = (a, w, b)
    return z, cache


def forward_act(a_previous, w, b, activation):
    if activation == 'relu':
        z, cache_lin = forward_lin(a_previous, w, b)
        a, cache_act = relu(z)
    elif activation == 'softmax':
        z, cache_lin = forward_lin(a_previous, w, b)
        a, cache_act = softmax(z)

    _cache = (cache_lin, cache_act)

    return a, _cache


def forward_prop(a_previous, params):

    h = len(params) // 2
    caches = []

    for i in range(1, h):
        a_previous, _cache = forward_act(a_previous, params['W' + str(i)],
                                         params['b' + str(i)], 'relu')
        caches.append(_cache)

    a_previous, _cache = forward_act(a_previous, params['W' + str(h)],
                                     params['b' + str(h)], 'softmax')
    caches.append(_cache)

    return a_previous, caches


def compute_cost(output, all_labels):

    # cost = -np.sum(all_labels*np.log(output) +
    #                (1 - all_labels)*np.log(1 - output))/len(all_labels)

    cost = -np.sum(all_labels*np.log(output)) / all_labels.shape[1]

    return cost


def backward_lin(dz, _cache):
    a_previous, w, b = _cache
    m = a_previous.shape[1]

    dw = (np.dot(dz, a_previous.T)) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    try:
        da_previous = np.dot(w.T, dz)
        return da_previous, dw, db
    except ValueError:
        return 0, dw, db


def backward_act(da_previous, _cache, labels, activation):

    linear_cache, activation_cache = _cache
    a, w, b = linear_cache
    z = activation_cache

    if activation == 'relu':
        dz = relu_back(da_previous, z)
        da_previous, dw, db = backward_lin(dz, linear_cache)
    else:
        dz = softmax_back(a, labels)
        da_previous, dw, db = backward_lin(dz, linear_cache)

    return da_previous, dw, db


def back_prop(a_previous, input_lab, _cache):

    grads = {}
    L = len(_cache)

    last_lin_cache, last_act_cache = _cache[-1]
    _cache = _cache[:-1]

    dz_prev = softmax_back(a_previous, input_lab)
    da_previous, dw, db = backward_lin(dz_prev, last_lin_cache)
    grads['dW{}'.format(L)] = dw
    grads['db{}'.format(L)] = db
    L = len(_cache)

    for i in reversed(range(L)):
        current_lin_cache, current_act_cache = _cache[i]
        dz_prev = relu_back(da_previous, current_act_cache)
        da_previous, dw, db = backward_lin(dz_prev, current_lin_cache)
        grads['dW{}'.format(i + 1)] = dw
        grads['db{}'.format(i + 1)] = db

    return grads


def update_parameters(params, grads, alpha):
    L = len(params) // 2

    for i in range(1, L + 1):
        params['W{}'.format(i)] -= alpha * grads['dW{}'.format(i)]
        params['b{}'.format(i)] -= alpha * grads['db{}'.format(i)]


def forw_back_upd(a_last, input_lab, params, alpha):
    a_prev, cache = forward_prop(a_last, params)
    cost = compute_cost(a_prev, input_lab)
    grads = back_prop(a_prev, input_lab, cache)
    update_parameters(params, grads, alpha)

    return cost


def calc_accuracy(a_final, inp_lab):

    nx = a_final.shape[0]
    m = a_final.shape[1]
    count = 0

    for x in range(m):
        prevision = one_hot2(a_final[:, x].reshape(nx, 1))
        label = inp_lab[:, x].reshape(nx, 1)

        if np.array_equal(prevision, label):
            count += 1

    return round(count / m * 100, 1)


def model_plain(input_set, input_lab, params, alpha, iterations):

    all_costs = []
    m = input_set.shape[1]

    for i in range(iterations):
        a_prev = input_set
        cost = forw_back_upd(a_prev, input_lab, params, alpha)
        all_costs.append(cost)

        if i % 3000 == 0:
            print("Cost at iteration {}: {}".format(i, round(cost, 4)))

    a_prev, cache = forward_prop(a_prev, params)
    accur = calc_accuracy(a_prev, input_lab)

    return accur


def model_mini_batch(input_set, input_lab, params, alpha, epochs, batch_size):
    all_cost = []
    mini_batches = input_set.shape[1]//batch_size + 1 if\
        input_set.shape[1] % batch_size else input_set.shape[1]//batch_size

    for i in range(epochs):
        cost = 0
        for x in range(mini_batches):
            inp = input_set[:, x * batch_size: (x + 1) * batch_size]
            label = input_lab[:, x * batch_size: (x + 1) * batch_size]
            cost += forw_back_upd(inp, label, params, alpha) / mini_batches
        all_cost.append(cost)

        if i % 3000 == 0:
            print("Cost at epoch {}: {}".format(i, round(cost, 4)))


hidden_layers = [64, 32, 16]
train_set, train_lab, dev_set, dev_lab, parameters = create_sets(hidden_layers)
# train_set, train_lab, parameters = create_practice_sets()

import time
start = time.time()
accuracy = model_plain(train_set, train_lab, parameters, 0.01, 6000)
# print('Accuracy for training set: {} %'.format(accuracy))
print(time.time() - start)
train_set, train_lab, dev_set, dev_lab, parameters = create_sets(hidden_layers)
start = time.time()
model_mini_batch(train_set, train_lab, parameters, 0.01, 6000, 512)
print(time.time() - start)
