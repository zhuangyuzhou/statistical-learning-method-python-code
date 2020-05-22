from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from decision_tree import DecisionTreeClassifier


class AdaBoostClassifier:

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.):

		"""
		AdaBoost Classfier
		
		Parameters
		----------
		base_estimator:
			The base estimator from which the boosted emsemble is built.
			If None, the base estimator is DecisionTreeClassifier(max_depth=1)
		n_estimators:
			The maximum number of estimators at which boosting is terminated.
			In case of perfect fit, the learning procedure is stopped early.
		learning_rate:
			Learning rate shrinks the contribution of each classifier by learning_rate.
		"""

		if base_estimator is None:
			base_estimator = DecisionTreeClassifier(max_depth=1)

		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate

	def fit(self, X, y, sample_weight=None):

		"""
		Build a boosted classifier from the training set (X, y)

		Parameters
		----------
		X:
			The training input samples of shape (n_samples, n_features)
		y:
			The target values (class labels) of shape (n_samples,)
		sample_weight:
			Sample weights of shape (n_samples,).
			If None, the sample weights are initialized to 1 / n_samples
		"""

		n_samples, n_features = X.shape
		classes = np.unique(y)
		n_classes = len(classes)

		if sample_weight is None:

			sample_weight = np.ones(n_samples) / n_samples

		estimators = []

		for i in range(self.n_estimators):

			estimator = deepcopy(self.base_estimator) # use a copy of the base estimator

			estimator.fit(X, y, sample_weight)

			error = estimator.predict(X) != y

			error_rate = np.sum(sample_weight * error)

			alpha = np.log((1 - error_rate) / error_rate) + np.log(n_classes - 1) # estimator coef

			estimators.append((estimator, alpha))

			if error_rate == 0: # perfect fit
				break

			sample_weight = sample_weight * np.exp(alpha * error) # update sample weights
			sample_weight = sample_weight / np.sum(sample_weight) # normalize

		self.classes_ = classes
		self.n_classes_ = n_classes
		self.estimators_ = estimators


	def predict(self, X):

		"""
		Predict classes for X.

		Parameters
		----------
		X : shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		y : ndarray of shape (n_samples,)
			The predicted classes.
		"""

		pred = np.zeros((X.shape[0], self.n_classes_))

		for estimator, alpha in self.estimators_:

			y = estimator.predict(X)

			for i, label in enumerate(self.classes_):
				pred[:, i] += alpha * (y == label)

		return self.classes_.take(np.argmax(pred, axis=1), axis=0)


	def score(self, X, y, sample_weight=None):
		
		"""
		Return the (weighted) mean accuracy on the given test data and labels
		X: Test samples, shape of (n_samples, n_features)
		y: True labels for X, shape of (n_samples,)
		"""

		return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':

	iris = load_iris() # load dataset
	X, y = iris.data, iris.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # choosing number of base estimators
	n_estimators_list = np.arange(1, 25, 2)
	train_acc = []
	test_acc = []

	for n_estimators in n_estimators_list:
		clf = AdaBoostClassifier(n_estimators=n_estimators)
		clf.fit(X_train, y_train)
		train_acc.append(clf.score(X_train, y_train))
		test_acc.append(clf.score(X_test, y_test))
	
	plt.figure()
	plt.plot(n_estimators_list, train_acc, label='training accuracy')
	plt.plot(n_estimators_list, test_acc, label='test accuracy')
	plt.xlabel('number of estimators')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()