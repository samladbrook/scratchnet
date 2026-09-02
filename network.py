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





