import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from model import CRF


class Args:
	def __init__(self):
		self.n_epochs = 3
		self.batch_size = 64
		self.learning_rate = 0.01
		self.weight_decay = 1e-4
		self.log_interval = 10


def load_dataset():

	X_train = np.load('data/X_train.npy', allow_pickle=True)
	X_test = np.load('data/X_test.npy', allow_pickle=True)
	y_train = np.load('data/y_train.npy', allow_pickle=True)
	y_test = np.load('data/y_test.npy', allow_pickle=True)

	return X_train, X_test, y_train, y_test


def train(model, X, y, optimizer, epoch, args):
	
	model.train()

	# random permutation of indices
	permutation = torch.randperm(X.shape[0])
	
	for i in range(0, X.shape[0], args.batch_size):

		optimizer.zero_grad()

		indices = permutation[i:i+args.batch_size]
		X_batch, y_batch = X[indices], y[indices]
		y_pred = []

		batch_loss = Variable(torch.Tensor([0]))
		for obs_seq, state_seq in zip(X_batch, y_batch):
			y_pred.append(model.decode(obs_seq))
			loss = model.nll_loss(obs_seq, state_seq)
			batch_loss += loss
		batch_loss /= X_batch.shape[0]
		batch_acc = model.score(y_batch, y_pred)
		batch_loss.backward()
		optimizer.step()

		batch_idx = i // args.batch_size
		if batch_idx % args.log_interval == 0:
			print('Epoch: {} [{}/{} {:.0f}%]\tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
				epoch, i, X.shape[0], 100 * i / X.shape[0], batch_loss.item(), batch_acc))


def test(model, X, y):

	model.eval()

	y_pred = []
	test_loss = Variable(torch.Tensor([0]))

	with torch.no_grad():
		for obs_seq, state_seq in zip(X, y):
			y_pred.append(model.decode(obs_seq))
			loss = model.nll_loss(obs_seq, state_seq)
			test_loss += loss
		test_loss /= X.shape[0]

	test_acc = model.score(y, y_pred)
	print('Test loss: {:.4f}\tTest Accuracy: {:.4f}'.format(test_loss.item(), test_acc))


if __name__ == '__main__':

	X_train, X_test, y_train, y_test = load_dataset()

	args = Args()

	model = CRF(n_states=2, n_obs=6)
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	for epoch in range(args.n_epochs):
		train(model, X_train, y_train, optimizer, epoch, args)

	test(model, X_test, y_test)