import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM:
	"""
	support vector machine for binary classification
	"""

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
				 coef0=0.0, tol=1e-3, max_iter=-1):

		"""
		C: penalty parameter C of the error term, inverse of regularization
		kernel: kernel type used, one of 'linear', 'poly', 'rbf'
		degree: degree of the polynomial kernel function ('poly')
		gamma: kernel coefficient for 'rbf' and 'poly'
			   If gamma='scale' then 1 / (n_features * X.var()) will be used.
               If gamma='auto' then 1 / n_features will be used.
		coef0: independent term in the polynomial kernel function ('poly')
		tol: tolerance for stopping criterion
		max_iter: maximum number of iterations, or -1 for no limit
		"""
		self.C = C
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.tol = tol
		self.max_iter = max_iter

	def make_label(self, y):
		"""
		Build labels such that positive class is +1 and negative class is -1
		"""
		y_copy = y.copy()
		c1, c2 = self.classes_
		y_copy[y==c1] = 1
		y_copy[y==c2] = -1
		return y_copy	
	
	def compute_kernel_matrix(self, X, X1=None):
		"""
		Compute the kernel matrix
		if X1 is None, between X and itself
		otherwise between X and X1
		"""
		if X1 is None:
			X1 = X
		
		if self.kernel == 'linear':
			K = np.dot(X, X1.T)
		elif self.kernel == 'rbf':
			X_norm = np.sum(X ** 2, axis=-1)
			X1_norm = np.sum(X1 ** 2, axis=-1)
			K = np.exp(-self.gamma * (X_norm[:,None] + X1_norm[None,:] - 2 * np.dot(X, X1.T)))
		elif self.kernel == 'poly':
			K = np.power(self.gamma * np.dot(X, X1.T) + self.coef0, self.degree)

		return K

	def compute_g_vector(self, a, b):

		return np.dot(a * self.y, self.K) + b


	def smo_step(self, a, b):
		"""
		One update step in SMO algorithm
		----
		return:
			a: updated a
			b: updated b
			stop_flag: True if stopping criterion is met
		"""
		y = self.y
		g = self.compute_g_vector(a, b)
		E = g - y
		
		# select index of a1: k1
		index_sv =[] # index of support vectors
		index_non_sv = [] # index of non support vectors
		max_violation_index_sv = [] # index of samples with max violation in support vectors 
		max_violation_index_non_sv = [] # index of samples with max violation in non support vectors
		for i in range(len(a)):
			if a[i] == 0:
				index_non_sv.append(i)
				violation = 1 - y[i] * g[i]
				if violation > self.tol:
					max_violation_index_non_sv.append((i, violation))
			elif a[i] == self.C:
				index_non_sv.append(i)
				violation = y[i] * g[i] - 1
				if violation > self.tol:
					max_violation_index_non_sv.append((i, violation))
			else:
				index_sv.append(i)
				violation = abs(y[i] * g[i] - 1)
				if violation > self.tol:
					max_violation_index_sv.append((i, violation))

		if len(max_violation_index_sv) > 0:
			k1, v1 = max(max_violation_index_sv, key=lambda x:x[1])
		elif len(max_violation_index_non_sv) > 0:
			k1, v1 = max(max_violation_index_non_sv, key=lambda x:x[1])
		else:
			return a, b, True

		a1 = a[k1]
		
		# select index of a2: k2
		k2 = np.argmax(np.abs(E - E[k1]))
		a2 = a[k2]
		if y[k1] == y[k2]:
			L = max(0, a2 + a1 - self.C)
			H = min(self.C, a2 + a1)
		else:
			L = max(0, a2 - a1)
			H = min(self.C, self.C + a2 - a1)
		a2_uncut = a2 + y[k2] * (E[k1] - E[k2]) / (self.K[k1, k1] + self.K[k2, k2] - 2 * self.K[k1, k2])
		if a2_uncut > H:
			a2_new = H
		elif a2_uncut < L:
			a2_new = L
		else:
			a2_new = a2_uncut
		
		a2_change = abs(a2_new - a2)
		if a2_change < self.tol: # not enough change in a2
			for k2 in index_sv + index_non_sv:
				a2 = a[k2]
				if y[k1] == y[k2]:
					L = max(0, a2 + a1 - self.C)
					H = min(self.C, a2 + a1)
				else:
					L = max(0, a2 - a1)
					H = min(self.C, self.C + a2 - a1)
				a2_uncut = a2 + y[k2] * (E[k1] - E[k2]) / (self.K[k1, k1] + self.K[k2, k2] - 2 * self.K[k1, k2])
				if a2_uncut > H:
					a2_new = H
				elif a2_uncut < L:
					a2_new = L
				else:
					a2_new = a2_uncut
				a2_change = abs(a2_new - a2)
				if a2_change >= self.tol:
					break

		if a2_change < self.tol: # stop
			return a, b, True

		a1_new = a1 + y[k1] * y[k2] * (a2 - a2_new)

		# update a and b
		a[k1] = a1_new
		a[k2] = a2_new
		
		b1 = -E[k1] -y[k1] * self.K[k1, k1] * (a1_new - a1) - y[k2] * self.K[k2, k1] * (a2_new - a2) + b
		b2 = -E[k2] -y[k1] * self.K[k1, k2] * (a1_new - a1) - y[k2] * self.K[k2, k2] * (a2_new - a2) + b
		if 0 < a1_new < self.C:
			b = b1
		elif 0 < a2_new < self.C:
			b = b2
		else:
			b = (b1 + b2) / 2

		return a, b, False


	def fit(self, X, y):
		"""
		X: shape (n_samples, n_features)
		y: shape (n_samples,)
		"""
		self.classes_ = np.unique(y)
		if (len(self.classes_) != 2):
			raise ValueError('y must have two classes, got %d' % len(self.classes_))
		
		n_samples, n_features = X.shape
		
		if self.gamma == 'scale':
			self.gamma = 1 / (n_features * X.var())
		elif self.gamma == 'auto':
			self.gamma = 1 / n_features

		self.X = X
		self.y = self.make_label(y)
		self.K = self.compute_kernel_matrix(X)
		
		# initialization
		a = np.zeros(n_samples)
		b = 0
		
		n_iter = 0
		while True:
			n_iter += 1
			a, b, flag = self.smo_step(a, b)
			if flag == True:
				break
			if n_iter == self.max_iter:
				break

		self.a = a
		self.b = b
		self.n_iter_ = n_iter

		return self


	def predict(self, X):
		"""
		Predict class labels for samples in X
		X: shape (n_samples, n_features)
		"""

		m = len(X)
		K = self.compute_kernel_matrix(self.X, X)
		g = np.dot(self.a * self.y, K) + self.b
		y_pred = np.zeros_like(g, dtype=self.classes_.dtype)
		y_pred[g >= 0] = self.classes_[0]
		y_pred[g < 0] = self.classes_[1]

		return y_pred

	def score(self, X, y):

		return accuracy_score(y, self.predict(X))


if __name__ == '__main__':

	X, y = load_breast_cancer(return_X_y=True) # dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	print('training set size:', X_train.shape[0])
	print('test set size:', X_test.shape[0])

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
    
	plt.figure()
	kernels = ['linear', 'rbf', 'poly']
	iters = np.arange(50, 1000, 50)
	for kernel in kernels:
		test_acc = []
		for max_iter in iters:
			clf = SVM(kernel=kernel, max_iter=max_iter)
			clf.fit(X_train, y_train)
			test_acc.append(clf.score(X_test, y_test))
		plt.plot(iters, test_acc, label=kernel)
	
	plt.xlabel('# iterations')
	plt.ylabel('test accuracy')
	plt.legend()
	plt.tight_layout()
	plt.plot()