import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.hyperparams import Hyperparameters as hp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init_parameters(input_size, hidden_sizes, output_size, init_method='Xavier'):
    scales = []
    layers = [input_size] + hidden_sizes
    if 'Xavier' in init_method:
        for i in range(1, len(layers) + 1):
            scales.append(np.sqrt(2. / layers[i - 1]))

    if 'Xavier' in init_method:
        first_layer_scale = scales[0]
    elif 'Random' in init_method:
        first_layer_scale = 0.01
    else:
        first_layer_scale = 0.0

    parameters = dict()
    parameters['W1'] = np.random.randn(hidden_sizes[0], input_size) * first_layer_scale
    parameters['b1'] = np.zeros((hidden_sizes[0], 1))

    for l in range(len(hidden_sizes)):
        if len(hidden_sizes) - 1 == l:  # last layer
            parameters['W' + str(l + 2)] = np.random.randn(output_size, hidden_sizes[l]) * scales[l + 1]
            parameters['b' + str(l + 2)] = np.zeros((output_size, 1))
        else:
            parameters['W' + str(l + 2)] = np.random.randn(hidden_sizes[l + 1], hidden_sizes[l]) * scales[l + 1]
            parameters['b' + str(l + 2)] = np.zeros((hidden_sizes[l + 1], 1))
    return parameters


def apply_non_linearity(Z, activation_func):
    if 'tanh' in activation_func:
        return tanh_forward(Z)
    elif 'relu' in activation_func:
        return relu_forward(Z)
    else:
        raise AssertionError


def forward_propagation(X, parameters, activation_func):
    num_layers = int(len(parameters) / 2)
    cache = dict()
    cache['A0'] = X

    for l in range(1, num_layers):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = linear_forward(cache['A' + str(l - 1)], W, b)
        cache['A' + str(l)] = apply_non_linearity(Z, activation_func)

    Z = linear_forward(Z, parameters['W' + str(num_layers)], parameters['b' + str(num_layers)])
    cache['A' + str(num_layers)] = sigmoid_forward(Z)
    return cache


def gradient_descent(parameters, derivatives):
    num_layers = int(len(parameters) / 2)
    for l in range(1, num_layers + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - hp.learning_rate * derivatives['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - hp.learning_rate * derivatives['db' + str(l)]
    return parameters


def compute_l2_reg_backprop(parameters, lambda_reg, batch_size):
    l2_regs = {name: (lambda_reg / batch_size) * w for name, w in parameters.items()
               if 'W' in name}
    return l2_regs


def backward_activation(Z, activation_func):
    if 'tanh' in activation_func:
        return tanh_backward(Z)
    elif 'relu' in activation_func:
        return relu_backward(Z)
    else:
        raise AssertionError


def backward_propagation(X, parameters, cache, Y_true, activation_func, num_layers=3, lambda_reg=0.0):
    derivatives = dict()
    batch_size = Y_true.shape[1]

    if lambda_reg != 0.0:
        l2_regs = compute_l2_reg_backprop(parameters, lambda_reg, batch_size)
    else:
        l2_regs = {name: 0.0 for name, w in parameters.items()
                   if 'W' in name}
    # sigmoid layer
    dZ_out = cache['A' + str(num_layers + 1)] - Y_true  # 1 x m
    dW_out = 1. / batch_size * np.dot(dZ_out, cache['A' + str(num_layers)].T) + l2_regs[
        'W' + str(num_layers + 1)]  # A1 h x m
    db_out = 1. / batch_size * np.sum(dZ_out, axis=1, keepdims=True)  # (1,1)

    derivatives['dA' + str(num_layers + 1)] = dZ_out
    derivatives['dW' + str(num_layers + 1)] = dW_out
    derivatives['db' + str(num_layers + 1)] = db_out
    # relu layer
    for i in reversed(range(num_layers)):
        # derivatives['dA' + str(i + 1)] = np.dot(parameters['W' + str(i + 2)].T, derivatives['dA' + str(i + 2)]) * relu_backward(cache['A' + str(i + 1)]) (h x 1) x (1 x m) x (h x m) = (h x m)
        derivatives['dA' + str(i + 1)] = np.dot(parameters['W' + str(i + 2)].T, derivatives['dA' + str(i + 2)])
        # derivatives['dA' + str(i + 1)] = derivatives['dA' + str(i + 1)] * tanh_backward(cache['A' + str(i + 1)])
        derivatives['dA' + str(i + 1)] = derivatives['dA' + str(i + 1)] * backward_activation(cache['A' + str(i + 1)],
                                                                                              activation_func)
        derivatives['dW' + str(i + 1)] = 1. / batch_size * np.dot(derivatives['dA' + str(i + 1)],
                                                                  cache['A' + str(i)].T) + l2_regs[
                                             'W' + str(i + 1)]  # (h x m) x (m x n_x)
        derivatives['db' + str(i + 1)] = 1. / batch_size * np.sum(derivatives['dA' + str(i + 1)], axis=1,
                                                                  keepdims=True)  # (h,1)

    return derivatives


def linear_forward(X, W, b):
    return np.dot(W, X) + b


def relu_forward(Z):
    return np.maximum(Z, 0)


def relu_backward(Z):
    return 1. * (Z > 0)


def tanh_forward(Z):
    return np.tanh(Z)


def tanh_backward(Z):
    return 1 - np.power(np.tanh(Z), 2)


def sigmoid_forward(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_backward(Z):
    return sigmoid_forward(Z) * (1 - sigmoid_forward(Z))


def cross_entropy_cost(Y_pred, Y_actual, parameters, lambda_reg=0.0):
    batch_size = Y_actual.shape[1]

    l2_term = 0.0
    if lambda_reg != 0.0:
        l2_term = compute_l2_reg(parameters, lambda_reg, batch_size)
    #  np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    # logs = np.log(Y_pred) * Y_actual + (1 - Y_actual) * np.log(1 - Y_pred)
    logs = np.multiply(-np.log(Y_pred), Y_actual) + np.multiply(-np.log(1 - Y_pred), (1 - Y_actual))
    cost = 1. / batch_size * np.nansum(logs) + l2_term  # TODO why loss increases?
    return cost


def compute_l2_reg(parameters, lambda_reg, batch_size):
    """
    Performs L2 calculation: sum_over(each layer L) : lambda/2*batch_size * W(L)^2
    :param parameters:
    :param lambda_reg:
    :param batch_size:
    :return: l2 reguralized value.
    """
    l2_req = 1. / batch_size * lambda_reg / 2 * np.sum([np.sum(np.square(w)) for name, w in parameters.items()
                                                        if 'W' in name])
    return l2_req


def compute_dropout():
    pass


def predict(Y_pred):
    pred = np.copy(Y_pred)
    pred[pred >= 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    return pred


def norm_data(X: pd.DataFrame):
    mean = X.mean()
    var = X.var()
    X_norm = (X - mean) / var
    return X_norm


# @PlotDecorator('cost')
def train_nn(X, Y, parameters, method, activation_func, fold_id):
    logger.info('Training neural network for %s selection method...', method)
    tqdm_iter = tqdm(range(hp.num_epochs))
    costs = []
    for i in tqdm_iter:
        num_batches = X.shape[1] // hp.batch_size
        idxs = np.arange(X.shape[1])
        np.random.shuffle(idxs)
        X = X[:, idxs]
        Y = Y[:, idxs]
        for batch in range(num_batches + 1):
            x_batch = X[:, batch * hp.batch_size:(batch + 1) * hp.batch_size]
            y_batch = Y[:, batch * hp.batch_size:(batch + 1) * hp.batch_size]

            cache = forward_propagation(x_batch, parameters, activation_func)
            derivatives = backward_propagation(x_batch, parameters, cache, y_batch, activation_func,
                                               len(hp.hidden_sizes), lambda_reg=hp.lambda_reg)
            parameters = gradient_descent(parameters, derivatives)

        if i % 100 == 0:
            cache = forward_propagation(X, parameters, activation_func)  # should i freeze the parameter update?
            predictions = predict(cache['A' + str(hp.num_layers)])
            acc = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
            cost = cross_entropy_cost(cache['A' + str(hp.num_layers)], Y, parameters, lambda_reg=hp.lambda_reg)
            tqdm_iter.set_postfix(accuracy=acc, loss=cost, method=method, fold_id=fold_id)
            costs.append(cost)
    return parameters, costs


def test_nn(X, Y, parameters, method, activation_func):
    cache = forward_propagation(X, parameters, activation_func)
    predictions = predict(cache['A' + str(hp.num_layers)])
    # print('Pred' , predictions)
    # print('Y' , Y)
    acc = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    logger.debug('Test accuracy for [%s] method : %d', method, acc)
    return acc
