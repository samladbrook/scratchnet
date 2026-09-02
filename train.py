from data import load_mnist
from network import NeuralNetwork

X_train, y_train, X_test, y_test = load_mnist()

net = NeuralNetwork()
probs = net.forward(X_train[:5])

print("output shape:", probs.shape)
print("each row sums to:", probs.sum(axis=1))
print("probabilities for image 0:", probs[0])
print("its guess:", probs[0].argmax(), " actual:", y_train[0])