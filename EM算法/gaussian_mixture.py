import warnings

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import cluster


class GaussianMixture:
	"""
	Representation of a Gaussian mixture model probability distribution.
	This class allows to estimate the parameters of a Gaussian mixture
	distribution using the EM algorithm.

	For simplicity, this implementation only considers Gaussian mixture
	distribution with full covariance.

	For relevant equations, see
	https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture
	"""
	def __init__(self, n_components=1, tol=1e-3, max_iter=100,
				 n_init=1, init_params='kmeans', random_state=None):
		"""
		Parameters
		----------
		n_components : int, defaults to 1.
			The number of mixture components.
		
		tol : float, defaults to 1e-3.
			The convergence threshold. EM iterations will stop when the 
			lower bound average gain is below this threshold.
		
		max_iter : int, defaults to 100.
			The maximum number of EM iteration to perform.
		
		n_init : int, defaults to 1.
			The number of initializations to perform. The best results are kept.
		
		init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
			The method used to initialize the weights, the means and the precision.
		
		random_state : int, RandomState instance or None, defaults to None.
			Controls the random seed given to the method chosen to initialize
			the parameters and the generation of random samples from the fitted
			distribution.
			Pass an int for reproducible output across multiple function calls.
		"""

		self.n_components = n_components
		self.tol = tol
		self.max_iter = max_iter
		self.n_init = n_init
		self.init_params = init_params
		self.random_state = random_state


	def fit(self, X):
		"""
		Estimate model parameters with the EM algorithm.

		The method fits the model n_init times and sets the parameters with which 
		the model has the largest likelihood (or lower bound).

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)

		Returns
		-------
		self
		"""

		max_lower_bound = -np.inf
		self.converged_ = False

		n_samples, _ = X.shape

		for init in range(1, self.n_init + 1):

			self._initialize_parameters(X)

			lower_bound = -np.inf

			for n_iter in range(1, self.max_iter + 1):
				prev_lower_bound = lower_bound

				lower_bound, resp = self._e_step(X)
				self._m_step(X, resp)

				change = lower_bound - prev_lower_bound

				if abs(change) < self.tol:
					self.converged_ = True
					break

			if lower_bound > max_lower_bound:
				max_lower_bound = lower_bound
				best_params = self._get_parameters()
				best_n_iter = n_iter

		if not self.converged_:
			warnings.warn('Initializations did not converge.')


		self._set_parameters(best_params)
		self.n_iter_ = best_n_iter
		self.lower_bound_ = max_lower_bound

		return self

	
	def _initialize_parameters(self, X):

		n_samples, _ = X.shape

		if self.init_params == 'kmeans':
			resp = np.zeros((n_samples, self.n_components))
			label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
								   random_state=self.random_state).fit(X).labels_
			resp[np.arange(n_samples), label] = 1
		
		elif self.init_params == 'random':
			resp = np.random.rand(n_samples, self.n_components, random_state=self.random_state)
			resp /= resp.sum(axis=1, keepdims=True)

		else:
			raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

		self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)


	def _estimate_gaussian_parameters(self, X, resp):

		n_samples, n_features = X.shape

		nk = resp.sum(axis=0)
		weights = nk / n_samples
		means = np.dot(resp.T, X) / nk[:, np.newaxis]
		
		covariances = np.empty((self.n_components, n_features, n_features))
		for k in range(self.n_components):
			diff = X - means[k]
			covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]

		return weights, means, covariances


	def _get_parameters(self):
		return (self.weights_, self.means_, self.covariances_)


	def _set_parameters(self, params):

		self.weights_, self.means_, self.covariances_ = params


	def _compute_log_gaussian_prob(self, X, mean, covariance):
		"""
		Compute log Gaussian prob given mean and covariance
		"""
		diff = X - mean

		# the precision matrix, inverse of covariance
		precision = np.linalg.inv(covariance)

		# the determinant of covariance
		det = np.linalg.det(covariance)
		
		return -.5 * (self.n_components * np.log(2 * np.pi) + np.log(det) + np.sum(np.dot(diff, precision) * diff, axis=1))


	def _estimate_weighted_log_prob(self, X):
		"""
		Estimate the weighted log probabilities of each sample in X
		"""
		n_samples = X.shape[0]
		
		log_prob = np.empty((n_samples, self.n_components))

		for k in range(self.n_components):
			log_prob[:, k] = self._compute_log_gaussian_prob(X, self.means_[k], self.covariances_[k])

		return self.weights_ * log_prob


	def _e_step(self, X):
		"""
		E step in the EM algorithm

		Parameters
		----------
		X : array-lik, shape (n_samples, n_features)

		Returns
		-------
		log_prob_norm : float
			Mean of the logarithms of the probabilities of each sample in X

		resp : array, shape (n_samples, n_components)
			The posterior probabilities (or responsibilities) of each sample in X
		"""

		weighted_log_prob = self._estimate_weighted_log_prob(X)
		
		log_prob_norm = logsumexp(weighted_log_prob, axis=1)
		
		log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

		return np.mean(log_prob_norm), np.exp(log_resp)


	def _m_step(self, X, resp):

		self.weights_, self.means_, self.covariances_ = self._estimate_gaussian_parameters(X, resp)


	def score_samples(self, X):
		"""
		Compute the weighted log probabilities for each sample in X
		
		Parameters
		----------
		X : array-like, shape (n_samples, n_features)

		Returns
		-------
		log_prob : array, shape (n_samples,)
		"""
		return logsumexp(self._estimate_weighted_log_prob(X), axis=1)


if __name__ == '__main__':

	# Number of random samples to generate for each Gaussian
	n_samples = 300

	np.random.seed(0)

	shifted_gaussian = np.random.randn(n_samples, 2) + np.array([10, 10])

	expanded_gaussian = 2 * np.random.randn(n_samples, 2) + np.array([-5, -10])
	
	stretched_gaussian = np.dot(np.random.randn(n_samples, 2), np.array([[0, -0.7], [3.5, 0.7]]))

	# training data are generated from two Gaussian distributions
	X_train = np.vstack([shifted_gaussian, expanded_gaussian, stretched_gaussian])

	# fit a Gaussian mixure model with two components
	clf = GaussianMixture(n_components=3, n_init=5)
	clf.fit(X_train)

	# display predicted scores by the moddel as a contour plot
	x = np.linspace(-20, 20)
	y = np.linspace(-20, 20)
	X, Y = np.meshgrid(x, y)
	XX = np.array([X.ravel(), Y.ravel()]).T
	Z = -clf.score_samples(XX)
	Z = Z.reshape(X.shape)

	C = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
					levels=np.logspace(0, 3, 10))
	plt.colorbar(C, shrink=0.8, extend='both')
	plt.scatter(X_train[:, 0], X_train[:, 1], .8)
	plt.title('Negative log-likelihood predicted by a GMM')
	plt.tight_layout()
	plt.show()

