from data import load_mnist
from network import NeuralNetwork, one_hot, cross_entropy_loss

X_train, y_train, X_test, y_test = load_mnist()

net = NeuralNetwork()

# grab one small batch and try memorise it perfectly
X_batch = X_train[:64]
y_batch = y_train[:64]
y_batch_onehot = one_hot(y_batch)

for step in range(1001):
	net.forward(X_batch)
	net.backward(X_batch, y_batch_onehot)
	net.update(0.1)
	if step % 200 == 0:
		probs = net.forward(X_batch)
		loss = cross_entropy_loss(probs, y_batch_onehot)
		acc = (probs.argmax(axis=1) == y_batch).mean()
		print(f"step {step:4d}	loss {loss:.4f}	 batch accuracy {acc:.1%}")