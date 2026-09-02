from data import load_mnist

X_train, y_train, X_test, y_test = load_mnist()

print("X_train:", X_train.shape, X_train.dtype)   # (60000, 784) float32
print("y_train:", y_train.shape, y_train.dtype)   # (60000,) uint8
print("X_test: ", X_test.shape)                   # (10000, 784)
print("pixel range:", X_train.min(), "to", X_train.max())  # 0.0 to 1.0
print("first 10 labels:", y_train[:10])
print("label range:", y_train.min(), "to", y_train.max())  # 0 to 9