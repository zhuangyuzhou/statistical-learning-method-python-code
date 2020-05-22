import numpy as np

class MultinomialHMM:
	"""
	Hidden Markov Model with multinomial (discrete) emissions

	Attributes
	----------
	startprob_ : array, shape (n_components,)
		Initial state occupation distribution.
	transmat_ : array, shape (n_components, n_components)
		Matrix of transition probabilities between states.
	emissionprob_ : array, shape (n_components, n_features)
		Probability of emitting a given symbol when in each state.
	"""

	def __init__(self, n_components=1, random_state=None, n_iter=10, tol=0.01):
		"""
		Parameters
		----------
		n_components : int, defaults to 1.
			Number of hidden states.
		random_state : int, RandomState or None, defaults to None.
			Random seed used for random initialization of parameters.
		n_iter : int, defaults to 10.
			Maximum number of iterations to perform in the EM algorithm.
		tol : float, defaults to 0.01.
			Convergence threshold. EM will stop if the gain in log-likelihood
			is below this value.
		"""
		self.n_components = n_components
		self.random_state = random_state
		self.n_iter = n_iter
		self.tol = tol


	def score(self, X):
		"""
		Compute the probablity of X under the model
		using the forward algorithm.

		Parameters
		----------
		X : array-like, shape (n_samples,)
			Feature vector of the observation sequence.

		Returns
		-------
		prob : float
			Probability (likelihood) of X.
		"""
		X = np.array(X)
		n_samples = X.shape[0]

		# store forward probabilities
		fwdlattice = np.zeros((n_samples, self.n_components))

		fwdlattice[0] = self.startprob_ * self.emissionprob_[:, X[0]]

		for t in range(1, n_samples):
			fwdlattice[t] = np.dot(fwdlattice[t - 1], self.transmat_) * self.emissionprob_[:, X[t]]

		return np.sum(fwdlattice[-1])


	def decode(self, X):
		"""
		Find the most likely state sequence corresponding to the observation
		sequence X using the Viterbi algorithm.

		Parameters
		----------
		X : array-like, shape (n_samples,)
			Feature vector of the observation sequence.

		Returns
		-------
		prob : float
			Probability of the produced state sequence.
		state_sequence : array, shape (n_samples,)
			Labels for each sample from X obtained via the decoding algorithm.
		"""
		X = np.array(X)
		n_samples = X.shape[0]

		prob_lattice = np.zeros((n_samples, self.n_components))
		state_lattice = np.zeros((n_samples, self.n_components))

		# initialization
		prob_lattice[0] = self.startprob_ * self.emissionprob_[:, X[0]]

		# iterating
		for t in range(1, n_samples):
			state_prob = prob_lattice[t - 1] * self.transmat_.T
			prob_lattice[t] = np.max(state_prob, axis=1) * self.emissionprob_[:, X[t]]
			state_lattice[t] = np.argmax(state_prob, axis=1)

		prob = np.max(prob_lattice[-1])

		# backtracking
		state_sequence = np.empty(n_samples, dtype=int)
		state_sequence[-1] = np.argmax(prob_lattice[-1])

		for t in reversed(range(1, n_samples)):
			state_sequence[t - 1] = state_lattice[t][state_sequence[t]]

		return prob, state_sequence


if __name__ == '__main__':

	model = MultinomialHMM(n_components=2)
	
	model.startprob_ = np.array([0.6, 0.4])
	model.transmat_ = np.array([[0.7, 0.3],
								[0.4, 0.6]])
	model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
									[0.6, 0.3, 0.1]])

	X = [1, 2, 0]
	print('Probability of observation sequence %s:' % X)
	print(model.score(X)) # 0.03276
	
	prob, state_sequence = model.decode(X)
	print('Most likely state sequence:')
	print(state_sequence) # [0, 0, 1]
	print('with probability:')
	print(prob) # 0.01512