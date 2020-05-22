import numpy as np


def onehot(y, classes):

	output = np.zeros((len(y), len(classes)))
	for i in range(len(y)):
		output[i] = (classes == y[i])
	return output


def initialize(n_classes, n_features):
	"""
	initialize weights, biases, and B matrix for BFGS method
	"""

	w = np.random.rand(n_classes, n_features)
	b = np.zeros((n_classes, 1))
	B = np.identity(n_classes * (n_features + 1))

	return w, b, B


def update_params_with_line_search(X, y, w, b, g, p, cost_func, c=1e-4, v=0.5):
	"""
	Armijo-Goldstein line search
	"""
	pw = p[:w.shape[0]*w.shape[1]].reshape(w.shape)
	pb = p[w.shape[0]*w.shape[1]:].reshape(b.shape)
	
	J = cost_func(X, y, w, b)

	i = 0
	while True:

		alpha = v ** i
		w_next = w + alpha * pw
		b_next = b + alpha * pb
		J_next = cost_func(X, y, w_next, b_next)
		if J_next <= J + c * alpha * np.dot(p, g):
			break
		i += 1

	return w_next, b_next, J_next


def update_matrix(B, g_delta, w_delta, b_delta):

	g_delta = g_delta.reshape((-1, 1))
	w_delta = np.append(w_delta.flatten(), b_delta.flatten()).reshape((-1, 1)) # flatten params
	term1 = np.dot(g_delta, g_delta.T) / (np.dot(g_delta.T, w_delta) + 1e-12)
	term2 = - np.dot(np.dot(np.dot(B, w_delta), w_delta.T), B) / (np.dot(np.dot(w_delta.T, B), w_delta) + 1e-12)
	B += term1 + term2
	return B