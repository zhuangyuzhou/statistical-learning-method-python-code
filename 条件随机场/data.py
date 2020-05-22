import numpy as np
from sklearn.model_selection import train_test_split


def generate_sequence(n_samples, params):
	"""
	Generate sequence of n_samples from the model params
	"""
	obs_seq = np.zeros(n_samples, dtype=int)
	state_seq = np.zeros(n_samples, dtype=int)
	
	state_seq[0] = np.random.choice([0, 1], p=params['initial_prob'])
	for i in range(1, n_samples):
		state_seq[i] = np.random.choice([0, 1], p=params['trans_mat'][state_seq[i - 1]])

	for i in range(n_samples):
		obs_seq[i] = np.random.choice(np.arange(6), p=params['emission_prob'][state_seq[i]])
	
	return obs_seq, state_seq


def generate_dataset(n_sequences, n_samples, params):

	X = np.empty((n_sequences, n_samples), dtype=int)
	y = np.empty((n_sequences, n_samples), dtype=int)

	for i in range(n_sequences):
		X[i], y[i] = generate_sequence(n_samples, params)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	return X_train, X_test, y_train, y_test


if __name__ == '__main__':

	np.random.seed(0)

	# initial prob {fair dice, loaded dice}
	initial_prob = [0.5, 0.5]

	# transition matrix
	trans_mat = [[0.6, 0.4],
				 [0.3, 0.7]]

	# emission prob
	# probabilities of rolling 1-6 for fair dice and loaded dice
	emission_prob = [[1/6,  1/6,  1/6,  1/6,  1/6,  1/6],
					 [0.04, 0.04, 0.04, 0.04, 0.04, 0.8]]

	params = {'initial_prob': initial_prob,
			  'trans_mat': trans_mat,
			  'emission_prob': emission_prob}

	# generate dataset
	X_train, X_test, y_train, y_test = generate_dataset(5000, 15, params)
	print('training set size:', X_train.shape[0])
	print('test set size:', X_test.shape[0])

	X_train.dump('data/X_train.npy')
	X_test.dump('data/X_test.npy')
	y_train.dump('data/y_train.npy')
	y_test.dump('data/y_test.npy')