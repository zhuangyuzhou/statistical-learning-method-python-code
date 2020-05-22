import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    
    def __init__(self, lr=0.1, max_iter=None):
        
        self.lr = lr # learning rate
        self.max_iter = max_iter
        self.index = 0
   
    
    def fetch_next(self):
        # fetch the next misclassified sample
        found = False
        for i in range(self.index, self.index + self.N):
            if self.y[i % self.N] * (np.dot(self.w, self.X[i % self.N]) + self.b) <= 0:
                found = True
                break
        
        if found:
            self.index = (i + 1) % self.N
            return self.X[i % self.N], self.y[i % self.N]
        else:
            return None, None
    
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y
        self.N = self.X.shape[0]
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        self.n_iter = 0
        while True:
            
            if self.n_iter >= self.max_iter:
                break
            
            xi, yi = self.fetch_next()
            if xi is None:
                break
            self.w += self.lr * xi * yi
            self.b += self.lr * yi
            self.n_iter += 1
              

def load_data():
    
    X = np.load('X.npy', allow_pickle=True)
    y = np.load('y.npy', allow_pickle=True)
    return X, y


def visualize(X, y, w, b):
    
    plt.figure()
    X_1 = X[y==1] # positive samples
    X_2 = X[y==-1] # negative samples
    plt.plot(X_1[:, 0], X_1[:, 1], 'bo', label='Class 1')
    plt.plot(X_2[:, 0], X_2[:, 1], 'go', label='Class 2')
    x = np.array([-5, 5])
    y = - (b + w[0] * x) / w[1]
    plt.plot(x, y, 'red', label='Decision boundary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    X, y = load_data()
    
    clf = Perceptron(lr=0.1, max_iter=300)
    clf.fit(X, y)
    print('number of iterations:', clf.n_iter)
    
    visualize(X, y, clf.w, clf.b)