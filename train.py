from data import load_mnist
from network import NeuralNetwork, one_hot, cross_entropy_loss

X_train, y_train, X_test, y_test = load_mnist()

net = NeuralNetwork()
batch_X = X_train[:64]
batch_y = y_train[:64]

probs = net.forward(batch_X)
loss = cross_entropy_loss(probs, one_hot(batch_y))

print("loss on untrained network:", round(loss, 4)) # should be around 2.3