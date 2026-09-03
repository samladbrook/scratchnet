import numpy as np
from data import load_mnist
from network import NeuralNetwork, one_hot, cross_entropy_loss, accuracy


# -----------------------
# training settings
EPOCHS = 10 			# how many passes over the training data 
BATCH_SIZE = 64			# how many images per step
LEARNING_RATE = 0.1 	# how big each backward step is

# -----------------------
# load the data and build the network
X_train, y_train, X_test, y_test = load_mnist()
net = NeuralNetwork()

# number of training images
num_samples = len(X_train)

# -----------------------
# training loop
# one epoch is one pass over the training data
for epoch in range(EPOCHS):
	# shuffle the training data
	order = np.random.permutation(num_samples)
	X_train = X_train[order]
	y_train = y_train[order]

	# go through the data in batches
	for start in range(0, num_samples, BATCH_SIZE):
		# get one batch of images and labels
		X_batch = X_train[start:start + BATCH_SIZE]
		y_batch = y_train[start:start + BATCH_SIZE]

		# 1. make predictions
		net.forward(X_batch)

		# 2. work out how the weights should change
		net.backward(X_batch, one_hot(y_batch))

		# 3. update the weights
		net.update(LEARNING_RATE)

	# check how the net did after each iteration
	predictions = net.forward(X_train)

	loss = cross_entropy_loss(predictions, one_hot(y_train))
	acc = accuracy(predictions, y_train)

	print(f"Epoch {epoch+1}/{EPOCHS} 	Loss: {loss:.4f}	Accuracy: {acc:.2%}")