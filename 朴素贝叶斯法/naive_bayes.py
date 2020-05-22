import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class NaiveBayes:
    
    def __init__(self, smooth=1):
        # smooth=1: Laplace smoothing
        self.smooth = smooth
    
    def fit(self, X, y):
        
        self.N = X.shape[0] # dataset size
        self.dims = X.shape[1] # feature dimensions
        
        prior = dict() # {y}
        conditional = dict() # {(dim, x, y)}
        
        label_values = set(y)
        for key in label_values:
        	prior[key] = (np.sum(y == key) + self.smooth) / (self.N + len(label_values) * self.smooth)

        for dim in range(self.dims):
        	feature_values = set(X[:, dim])
        	for a in feature_values:
        		for c in label_values:
        			conditional[(dim, a, c)] = (np.sum((X[:, dim] == a) & (y == c)) + self.smooth) / (np.sum(y == c) + len(feature_values) * self.smooth)

        self.labels = list(label_values)
        self.prior = prior
        self.conditional = conditional
    
    def predict(self, x):
        
        probs = []
        for label in self.labels:
            prob = self.prior[label]
            for dim in range(self.dims):
                key = (dim, x[dim], label)
                prob *= self.conditional.get(key, 1 / len(set(X[:, dim])))
            probs.append(prob)
        
        probs = np.array(probs)
        y = self.labels[np.argmax(probs)] # choose the label with max prob
        return y, max(probs)
        

def generate_data():

	X1 = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]).reshape((1,-1))
	X2 = np.array([0,1,1,0,0,0,1,1,2,2,2,1,1,2,2]).reshape((1,-1))
	y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    
	X = np.append(X1, X2, axis=0).T
	return X, y


def visualize(X, y):
    
    plt.figure()
    plt.plot(X[y==-1][:,0], X[y==-1][:,1], 'o', alpha=0.5, label='label -1')
    plt.plot(X[y==1][:,0], X[y==1][:,1], 'o', alpha=0.5, label='label 1')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    X, y = generate_data()
    visualize(X, y)

    clf = NaiveBayes(smooth=1) # Naive Bayes with Laplace smoothing
    clf.fit(X, y)
    
    print(clf.predict([2, 0]))
	