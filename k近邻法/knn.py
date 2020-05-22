import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import Counter


# Node of KD tree
class Node:
	def __init__(self, data, label, sp_dim, left=None, right=None):
		self.data = data
		self.label = label
		self.sp_dim = sp_dim
		self.left = left
		self.right = right


class KDTree:
    
    def __init__(self, X, y):

    	self.k = X.shape[1] # dimension of kd tree
    	dataset = list(zip(X, y))
    	self.root = self.build_tree(dataset, 0) # root node of kd tree

    def build_tree(self, dataset, sp_dim):

    	if len(dataset) == 0:
    		return

    	# sort the dataset according to the sp_dim
    	dataset.sort(key = lambda x : x[0][sp_dim])

    	# find the median point
    	median_index = len(dataset) // 2
    	node = Node(dataset[median_index][0], dataset[median_index][1], sp_dim)

    	# build left and right nodes
    	next_sp_dim = (sp_dim + 1) % self.k
    	node.left = self.build_tree(dataset[:median_index], next_sp_dim)
    	node.right = self.build_tree(dataset[median_index + 1:], next_sp_dim)

    	return node
    
    # predict label for a given sample
    def predict(self, x, k, p):

    	knn = self.search_knn(x, k, p)
    	X_knn = np.array([x.data for x in knn])
    	y_knn = np.array([x.label for x in knn])
    	y_pred = max(Counter(y_knn).items(), key=lambda x:x[1])[0]
    	return y_pred, X_knn


    # search for the k nearest neighbors (k-nn)
    def search_knn(self, x, k, p):
        
        # list storing tuples of (distance, node)
        # TODO: use heap implementation for faster push and pop
        knn = [(np.inf, None)] * k
        self.visit(x, self.root, knn, p)
        knn = np.array([x[1] for x in knn])
        return knn


    def visit(self, x, node, knn, p):

    	if node is None:
    		return

    	# calculate the distance to the separation hyperplane
    	dist = x[node.sp_dim] - node.data[node.sp_dim]

    	# iterate down the tree
    	self.visit(x, node.left, knn, p) if dist < 0 else self.visit(x, node.right, knn, p)

    	# calculate the current distance
    	cur_dist = np.linalg.norm(x - node.data, p) # L_p norm
    	if cur_dist < max(knn)[0]:
    		knn.append((cur_dist, node))
    		knn.remove(max(knn))

    	# if dist < max dist in knn
    	# visit the other child node
    	if dist < max(knn)[0]:
    		self.visit(x, node.right, knn, p) if dist < 0 else self.visit(x, node.left, knn, p)


def generate_data():
    X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.array([0, 0, 1, 0, 1, 1])
    return X, y

def visualize(X, y):
    
    plt.plot(X[y==0].T[0], X[y==0].T[1], 'bo', label='Class 1')
    plt.plot(X[y==1].T[0], X[y==1].T[1], 'go', label='Class 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    X, y = generate_data()
    visualize(X, y)

    kdtree = KDTree(X, y)
    y_pred, X_knn = kdtree.predict([4, 3], k=2, p=2)
    print('predicted label:', y_pred)
    print('k nearest neighbors:', X_knn, sep='\n')