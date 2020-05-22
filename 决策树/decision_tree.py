import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Node for decision tree
class TreeNode:
	def __init__(self, X, y, sample_weight, depth):

		self.X = X
		self.y = y
		self.total_weight = np.sum(sample_weight) # relative to all samples
		self.sample_weight = sample_weight / self.total_weight # relative to current node
		self.depth = depth
		self.gini = None # gini index at this node
		# for internal nodes
		self.split_feature = None # splitting feature
		self.split_point = None
		self.left = None
		self.right = None
		# for leaf nodes
		self.label = None


# Mimic the sklearn interface of DecisionTreeClassifier
class DecisionTreeClassifier:

	def __init__(self, criterion='gini',
				 max_depth=None,
				 min_samples_split=2,
				 min_impurity_decrease=0.0,
				 alpha=0.0):
		
		"""
		criterion: used to measure the quality of a split (only 'gini' supported right now) 
		max_depth: The maximum number depth of the tree. If None, no limitation.
		min_samples_split: the minimum number of samples required to split an internal node
		min_impurity_decrease: the minimum impurity decrease required for split
		alpha: complexity parameter used for CART pruning
		"""

		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_impurity_decrease = min_impurity_decrease
		self.alpha = alpha


	def compute_gini(self, y, sample_weight):

		gini = 1
		for label in self.labels:
			gini -= np.sum(sample_weight * (y==label)) ** 2
		return gini
	
	
	def compute_conditional_gini(self, y, sample_weight, l_slice, r_slice):

		l_sample_weight, r_sample_weight = sample_weight[l_slice], sample_weight[r_slice]
		l_total_weight, r_total_weight = np.sum(l_sample_weight), np.sum(r_sample_weight)
		return l_total_weight * self.compute_gini(y[l_slice], l_sample_weight / l_total_weight) + r_total_weight * self.compute_gini(y[r_slice], r_sample_weight / r_total_weight)


	def get_most_common_label(self, y, sample_weight):
		"""
		get weighted most common label
		y: array of labels, shape (n_samples,)
		sample_weight: array of sample weights, shape (n_samples,)
		"""

		probs = np.zeros(len(self.labels))

		for i, label in enumerate(self.labels):
			probs[i] = np.sum(sample_weight * (y == label))
		
		return self.labels[np.argmax(probs)]


	def construct(self, node):

		"""
		recursively construct the tree according to the criterion
		"""
		
		N = node.X.shape[0] # sample size at current node
		node.gini = self.compute_gini(node.y, node.sample_weight)
		if N < self.min_samples_split or node.depth == self.max_depth:
			node.label = self.get_most_common_label(node.y, node.sample_weight)
			return

		# find the best feature and split point to split
		argsort = np.argsort(node.X, axis=0)
		gini_memo = [] # list of tuples (feature, split_index, gini_index)
		for feat in range(self.n_features):
			conditional_gini_list = []
			for i in range(1, N):
				conditional_gini_list.append(self.compute_conditional_gini(node.y, node.sample_weight, argsort[:i, feat], argsort[i:, feat]))
			split_index = np.argmin(conditional_gini_list) + 1
			gini_memo.append((feat, split_index, min(conditional_gini_list)))

		best_feature, best_split_index, best_gini = min(gini_memo, key=lambda x:x[2])
		
		if node.gini - best_gini < self.min_impurity_decrease:
			node.label = self.get_most_common_label(node.y, node.sample_weight)
			return
		
		node.split_feature = best_feature
		node.split_point = node.X[argsort[best_split_index, best_feature], best_feature]
		l_slice, r_slice = argsort[:best_split_index, best_feature], argsort[best_split_index:, best_feature]
		node.left = TreeNode(node.X[l_slice], node.y[l_slice], node.sample_weight[l_slice], node.depth + 1)
		node.right = TreeNode(node.X[r_slice], node.y[r_slice], node.sample_weight[r_slice], node.depth + 1)
		self.construct(node.left)
		self.construct(node.right)


	def prune(self, node):

		if node.label is not None: # leaf node
			node.cost = len(node.y) * node.gini + self.alpha
			return

		self.prune(node.left)
		self.prune(node.right)

		# whether or not to prune the current node into a leaf node
		cost_before_pruning = (node.left.total_weight * node.left.cost + node.right.total_weight * node.right.cost) / node.total_weight
		cost_after_pruning = len(node.y) * node.gini + self.alpha
		if cost_after_pruning <= cost_before_pruning:
			node.cost = cost_after_pruning
			node.label = self.get_most_common_label(node.y, node.sample_weight)
			node.left = None
			node.right = None
		else:
			node.cost = cost_before_pruning


	def fit(self, X, y, sample_weight=None):

		"""
		build a decision tree classifier from the training set (X, y)
		X: shape of (n_samples, n_features)
		y: shape of (n_samples,)
		sample_weight: Sample weights of shape (n_samples,)
		"""

		n_samples, self.n_features = X.shape

		if sample_weight is None:
			sample_weight = np.ones(n_samples) / n_samples

		self.labels = np.unique(y)
		root = TreeNode(X, y, sample_weight, 0) # root of the decision tree
		self.construct(root) # construct the full decision tree
		self.prune(root) # pruning given alpha value
		self.root = root

		return self


	def predict(self, X):

		pred = []
		for x in X:
			node = self.root
			while node.label is None:
				if x[node.split_feature] < node.split_point:
					node = node.left
				else:
					node = node.right

			pred.append(node.label)

		return pred

	def score(self, X, y):
		
		"""
		return the mean accuracy on the given test data and labels
		X: test samples, shape of (n_samples, n_features)
		y: true labels of X, shape of (n_samples,)
		"""
		
		return accuracy_score(y, self.predict(X))


if __name__ == '__main__':

	iris = load_iris() # dataset
	X, y = iris.data, iris.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # effect of alpha
	alpha_list = np.arange(0, 5, 0.5)
	train_acc = []
	test_acc = []

	for alpha in alpha_list:
		clf = DecisionTreeClassifier(alpha=alpha)
		clf.fit(X_train, y_train)
		train_acc.append(clf.score(X_train, y_train))
		test_acc.append(clf.score(X_test, y_test))
	
	plt.figure()
	plt.plot(alpha_list, train_acc, label='training accuracy')
	plt.plot(alpha_list, test_acc, label='test accuracy')
	plt.xlabel('alpha value')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()

	# effect of max_depth
	max_depth_list = np.arange(1, 10)
	train_acc = []
	test_acc = []

	for max_depth in max_depth_list:
		clf = DecisionTreeClassifier(max_depth=max_depth)
		clf.fit(X_train, y_train)
		train_acc.append(clf.score(X_train, y_train))
		test_acc.append(clf.score(X_test, y_test))
	
	plt.figure()
	plt.plot(max_depth_list, train_acc, label='training accuracy')
	plt.plot(max_depth_list, test_acc, label='test accuracy')
	plt.xlabel('max depth')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()
