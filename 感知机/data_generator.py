# generate training data for perceptron algorithm
# training data is guaranteed to be linearly separable

import numpy as np
import matplotlib.pyplot as plt

class Config:
	def __init__(self):
		# define the separating hyperplane: y=w*x+b
		self.w = np.array([1, 2])
		self.b = 3
		self.N = 100 # number of training samples



def visualize(X, y, w, b):
    
    plt.figure()
    X_1 = X[y==1] # positive samples
    X_2 = X[y==-1] # negative samples
    plt.plot(X_1[:, 0], X_1[:, 1], 'bo')
    plt.plot(X_2[:, 0], X_2[:, 1], 'go')
    x = np.array([-5, 5])
    y = - (b + w[0] * x) / w[1]
    plt.plot(x, y, 'red')
    plt.show()


if __name__ == '__main__':

	config = Config()
	data_X = []
	data_y = []

	for _ in range(config.N):

		x = -5 + 10 * np.random.rand(2)
		out = np.dot(config.w, x) + config.b
		y = 1 if out > 0 else -1
		data_X.append(x)
		data_y.append(y)

	data_X = np.array(data_X)
	data_y = np.array(data_y)

	# visualize
	visualize(data_X, data_y, config.w, config.b)
    
    # save
	data_X.dump('X.npy')
	data_y.dump('y.npy')