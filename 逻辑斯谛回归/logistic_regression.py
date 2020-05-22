import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import *


class LogisticRegression:

	def __init__(self, penalty='l2', tol=1e-4, C=1.0,
				 random_state=None, solver='bfgs', max_iter=100):
		"""
		penalty: the norm used in regularization
		tol: tolerance for stopping criteria
		C: inverse of regularization strength
		random_state: seed for random initialization
		solver: algorithm to use in the optimization problem
		max_iter: maximum number of iterations taken for the solvers to converge
		"""

		# check solver
		all_solvers = ['bfgs']
		if solver not in all_solvers:
			raise ValueError("Logistic Regression supports only solvers in %s, got"
							 " %s." % (all_solvers, solver))

		# check penalty
		all_penalties = ['l2']
		if penalty not in all_penalties:
			raise ValueError("Logistic Regression supports only penalties in %s, got"
							 " %s." % (all_penalties, penalty))

		self.penalty = penalty
		self.tol = tol
		self.C = C
		self.random_state = random_state
		self.solver = solver
		self.max_iter = max_iter

	
	def compute_cost(self, X, y, w, b):
		"""
		X: shape (n_samples, n_features)
		y: shape (n_samples, n_classes)
		w: shape (n_classes, n_features)
		b: shape (n_classes, 1)
		"""
		m = len(X)
		z = np.dot(w, X.T) + b # shape (n_classes, n_samples)
		a = np.exp(z)
		a = a / np.sum(a, axis=0) # shape (n_classes, n_samples)
		
		if self.penalty == 'l2':
			J = (-self.C * np.sum(y.T * np.log(a)) + np.sum(w ** 2)) / m

		return J

	
	def compute_gradient(self, X, y, w, b):
		"""
		compute gradient in logistic regression
		X: shape (n_samples, n_features)
		y: shape (n_samples, n_classes)
		w: shape (n_classes, n_features)
		b: shape (n_classes, 1)
		"""

		m = len(X)
		z = np.dot(w, X.T) + b # shape (n_classes, n_samples)
		a = np.exp(z)
		a = a / np.sum(a, axis=0) # shape (n_classes, n_samples)
		
		if self.penalty == 'l2':
			dw = (self.C * np.dot(a - y.T, X) + 2 * w) / m
			db = self.C * np.sum(a - y.T, axis=1, keepdims=True) / m

		return dw, db


	def fit(self, X, y):

		self.classes_ = np.unique(y)
		n_samples, n_features = X.shape
		n_classes = len(self.classes_)

		y = onehot(y, self.classes_) # use one-hot encoding
		n_iter = 0 # actual number of iterations
		costs = [] # cost computed at each iteraction

		if self.solver == 'bfgs':

			np.random.seed(self.random_state)
			w, b, B = initialize(n_classes, n_features)
			dw, db = self.compute_gradient(X, y, w, b)
			g = np.append(dw.flatten(), db.flatten())

			while True:

				n_iter += 1
				p = np.linalg.solve(B, -g)
				w_next, b_next, J_next = update_params_with_line_search(X, y, w, b, g, p, self.compute_cost)
				costs.append(J_next)
				dw, db = self.compute_gradient(X, y, w_next, b_next)
				g_next = np.append(dw.flatten(), db.flatten())
				
				if np.linalg.norm(g_next) < self.tol:
					break
				
				B = update_matrix(B, g_next - g, w_next - w, b_next - b)
				g, w, b = g_next, w_next, b_next

				if n_iter == self.max_iter:
					break
			
			self.w = w
			self.b = b

		self.n_iter_ = n_iter
		self.costs_ = costs

		return self

	
	def predict(self, X):
		"""
		Predict class labels for samples in X
		X: shape (n_samples, n_features)
		"""

		m = len(X)
		z = np.dot(self.w, X.T) + self.b
		a = np.exp(z)
		probs = a / np.sum(a, axis=0) # shape (n_classes, n_samples)
		indices = probs.argmax(axis=0) # indices of largest probs

		return self.classes_[indices]

	def score(self, X, y):

		return accuracy_score(y, self.predict(X))



if __name__ == '__main__':

	iris = load_iris() # dataset
	X, y = iris.data, iris.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	clf = LogisticRegression(C=2.0, max_iter=100, random_state=0)
	clf.fit(X_train, y_train)
	print('number of iterations:', clf.n_iter_)
	print('training accuracy:', clf.score(X_train, y_train))
	print('test accuracy:', clf.score(X_test, y_test))

	plt.plot(clf.costs_)
	plt.xlabel('# iteration')
	plt.ylabel('loss')
	plt.title('Loss curve')
	plt.show()
