import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class CRF(nn.Module):

	def __init__(self, n_states, n_obs):
		"""
		Parameters
		----------
		n_states : number of possible states
		n_obs : number of possible observations
		"""
		super().__init__()

		self.n_states = n_states
		self.n_obs = n_obs

		# initialize weights
		self.start = nn.Parameter(torch.empty(n_states))
		self.transition = nn.Parameter(torch.empty(n_states, n_states))
		self.emission = nn.Parameter(torch.empty(n_states, n_obs))

		nn.init.normal_(self.start, -1, 0.1)
		nn.init.normal_(self.transition, -1, 0.1)
		nn.init.normal_(self.emission, -1, 0.1)


	def _compute_log_numerator(self, obs_seq, state_seq):
		"""
		Compute log of numerator (unnormalized prob)

		Parameters
		----------
		obs_seq : array, shape (n_samples,)
		state_seq : array, shape (n_samples,)

		Returns
		-------
		score : Tensor, scalar
		"""
		n_samples = obs_seq.shape[0]

		score = Variable(torch.Tensor([0]))
		score += self.start[state_seq[0]] + self.emission[state_seq[0], obs_seq[0]]
		for i in range(1, n_samples):
			score += self.transition[state_seq[i - 1], state_seq[i]] + self.emission[state_seq[i], obs_seq[i]]
		return score


	def _compute_log_denominator(self, obs_seq):
		"""
		Compute log of denominator (normalization factor) using the forward algorithm.

		Parameters
		----------
		state_seq : array, shape (n_samples,)

		Returns
		-------
		score : Tensor, scalar
		"""
		n_samples = obs_seq.shape[0]

		alpha = self.start + self.emission[:, obs_seq[0]]
		for i in range(1, n_samples):
			alpha = torch.log(torch.mm(torch.exp(alpha).unsqueeze(0), torch.exp(self.transition))).view(-1) + self.emission[:, obs_seq[i]]

		# TODO: numerically stable logsumexp
		return torch.log(torch.sum(torch.exp(alpha)))



	def nll_loss(self, obs_seq, state_seq):
		"""
		Compute the negative log likelihood loss for the given observation sequence
		and state sequence.

		Parameters
		----------
		obs_seq : ndarray, shape (n_samples,)
		state_seq : ndarray, shape (n_samples,)
		"""
		log_numerator = self._compute_log_numerator(obs_seq, state_seq)
		log_denominator = self._compute_log_denominator(obs_seq)
		return log_denominator - log_numerator


	def decode(self, obs_seq):
		"""
		Find the most likely state sequence estimated from the learned parameters
		using the Viterbi algorithm.

		Parameters
		----------
		obs_seq : array, shape (n_samples,)

		Returns
		-------
		state_seq : array, shape (n_samples,)
		"""
		n_samples = obs_seq.shape[0]

		start = self.start.detach().numpy()
		transition = self.transition.detach().numpy()
		emission = self.emission.detach().numpy()

		# unnormalized probabilities
		prob_lattice = np.empty((n_samples, self.n_states), dtype=int)
		state_lattice = np.empty((n_samples, self.n_states), dtype=int)

		prob_lattice[0] = start + emission[:, obs_seq[0]]

		for i in range(1, n_samples):
			state_prob = prob_lattice[i - 1] + transition.T
			prob_lattice[i] = np.max(state_prob, axis=1) + emission[:, obs_seq[i]]
			state_lattice[i] = np.argmax(state_prob, axis=1)

		# backtracking
		state_seq = np.empty(n_samples, dtype=int)
		state_seq[-1] = np.argmax(prob_lattice[-1])
		for i in reversed(range(1, n_samples)):
			state_seq[i - 1] = state_lattice[i][state_seq[i]]

		return state_seq


	def score(self, y_true, y_pred):

		accuracy = 0
		n_samples = 0
		for i in range(y_true.shape[0]):
			accuracy += np.sum(y_true[i] == y_pred[i])
			n_samples += len(y_true[i])
		accuracy /= n_samples

		return accuracy

