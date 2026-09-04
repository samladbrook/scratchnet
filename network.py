import numpy as np

def relu(z):
	"""ReLu activation function: keep the positive values and zero out the negative ones"""
	return np.maximum(0, z)


def softmax(z):
	"""Convert the network output scores into probabilities

	For example:
	[2.1, 0.5, 1.2] -> [0.63, 0.13, 0.24]

	All the probs will add up to 1
	"""

	# take off the largest value to stop np.exp() from returning
	# really large numbers
	z = z - np.max(z, axis=1, keepdims=True)

	# Convert the scores to a positive number
	exp_values = np.exp(z)

	# divide each value by the total so they add up to 1
	probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

	return probabilities


def one_hot(y, num_classes=10):
	"""Turn the labels like [3, 0] into rows with a 1 in the right column

	[3, 0] -> 	[[0,0,0,1,0,0,0,0,0,0],
				 [1,0,0,0,0,0,0,0,0,0]]			
	"""
	encoded = np.zeros((y.size, num_classes))
	encoded[np.arange(y.size), y] = 1.0
	return encoded


def cross_entropy_loss(probs, y_onehot):
	"""Average cross entropy loss over a batch

	Work out how wrong the prediction is if it is wrong"""
	n = probs.shape[0]
	clipped = np.clip(probs, 1e-12, 1.0)
	return -np.sum(y_onehot * np.log(clipped)) / n

def accuracy(probs, y_true):
	"""
	Fraction of predictions the network got right

	probs 	probabilities from forward()
	y_true 	the right digit labels
	"""
	# the networks guess is whichever class has the highest probability
	predictions = probs.argmax(axis=1)
	# compare all guesses to the truth and average
	return np.mean(predictions == y_true)

def confusion_matrix(y_true, y_pred, num_classes=10):
	"""
	Count how often each true digit was predicted as each digit

	rows = the actual digit
	cols = the predicted digit

	the diagonal is the correct guesses

	y_true	the correct labels
	y_pred	the predicted labels
	"""
	# start with a 10x10 matrix of zeros
	matrix = np.zeros((num_classes, num_classes), dtype=int)
	# for every (true, predicted) pair we add one to the matching cell
	np.add.at(matrix, (y_true, y_pred), 1)
	return matrix


class NeuralNetwork:
	def __init__(self, input_size=784, hidden_size=128, output_size=10, seed=0):

		# random number generator
		# using a seed means we get the same random weights each time
		rng = np.random.default_rng(seed)


		# -----------------------
		# Input -> hidden layer

		# Each input is connected to each hidden neuron
		# 784 inputs x 128 hidden neurons

		self.W1 = rng.standard_normal((input_size, hidden_size))
		self.W1 *= np.sqrt(2 / input_size)

		# each hidden neuron also has its own bias
		self.b1 = np.zeros(hidden_size)
		
		# -----------------------
		# Hidden -> output layer

		# 128 hidden neurons x 10 output neurons
		self.W2 = rng.standard_normal((hidden_size, output_size))
		self.W2 *= np.sqrt(2 / hidden_size)

		# one bias for each of the 10 output neurons
		self.b2 = np.zeros(output_size)

	def forward(self, X):
		"""
		Run the input data through the neural network.

		For MNIST

		X
		784 pixel values
			|
		Hidden layer
		128 neurons
			|
		Output layer
		10 digit classes
			|
		Probabilities
		"""

		# step 1
		# pass the input into the hidden layer
		self.z1 = X @ self.W1 + self.b1

		# step 2
		# apply ReLu to the hidden layer
		self.a1 = relu(self.z1)

		# step 3
		# pass the hidden layer into the output layer
		self.z2 = self.a1 @ self.W2 + self.b2

		# step 4
		# turn the output scores into probabilities
		self.a2 = softmax(self.z2)

		# return the final probabilities
		return self.a2

	def backward(self, X, y_true):
		"""
		Walk backwards through the network to measure how much each weight
		and bias contributed to the loss (the gradient)

		foward went: 	X -> z1 -> a1 -> z2 -> a2 -> loss
		backwards goes:	loss -> a2 -> z2 -> a1 -> z1 -> X
		(we find the gradient at each step)

		X 		784 pixel values we just ran forward
		y_true 	the one hot correct answers
		"""
		# How many images are in the batch
		n = X.shape[0]

		# -----------------------
		# Output layer gradients
		# softmax + cross entropy make this the error
		# dividing by n turns the total error into the average
		dz2 = (self.a2 - y_true) / n

		# how each hidden -> output weight affected the loss
		self.dW2 = self.a1.T @ dz2
		# how each output bias affected the loss
		self.db2 = np.sum(dz2, axis=0)

		# -----------------------
		# Hidden layer gradients
		# push the error back across W2 to make sure we reach the hidden neurons
		da1 = dz2 @ self.W2.T
		# a hidden neuron only passed signal if ReLu let it through
		# this means the gradient only flows back through those same neurons
		dz1 = da1 * (self.z1 > 0)

		# how each input -> hidden weight affected the loss
		self.dW1 = X.T @ dz1
		# how each hidden bias affected the loss
		self.db1 = np.sum(dz1, axis=0)

	def update(self, learning_rate):
		"""
		Take one step downhill: bump every parametre a little bit
		in the direction that lowers the loss / gradient

		learning_rate = how big a step to take
		"""
		# subtract a fraction of each gradient to move against it
		self.W1 -= learning_rate * self.dW1
		self.b1 -= learning_rate * self.db1
		self.W2 -= learning_rate * self.dW2
		self.b2 -= learning_rate * self.db2






