# main file, handles training and the extra reports
#
#   python scratchnet.py                        train, print test accuracy
#   python scratchnet.py --show-full-stats      per digi + confusion matrix
#   python scratchnet.py --show-missed digits   draw some mistakes
#   python scratchnet.py --show-lr-graph        graph loss vs lr before training
#   python scratchnet.py --lr 0.15 --epochs 15 --save model.npz
import argparse

import numpy as np
from data import load_mnist
from network import NeuralNetwork, one_hot, cross_entropy_loss, accuracy
import report


def train(net, X_train, y_train, epochs, batch_size, learning_rate, verbose=True):
    """
    Train the network with gradient descent

    net                 the network to train
    X_train, y_train    the training images and labels
    epochs              how many passes over the data
    batch_size          images per step
    learning_rate       step size
    verbose             print the loss and accuracy per epoch
    """
    # number of training images
    num_samples = len(X_train)

    # -----------------------
    # training loop
    # one epoch is one pass over the training data
    for epoch in range(epochs):
        # shuffle the training data
        order = np.random.permutation(num_samples)
        X_train = X_train[order]
        y_train = y_train[order]

        # go through the data in batches
        for start in range(0, num_samples, batch_size):
            # get one batch of images and labels
            X_batch = X_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]

            # 1. make predictions
            net.forward(X_batch)

            # 2. work out how the weights should change
            net.backward(X_batch, one_hot(y_batch))

            # 3. update the weights
            net.update(learning_rate)

        # report progress at the end of each step
        if verbose:
            probs = net.forward(X_train)
            loss = cross_entropy_loss(probs, one_hot(y_train))
            acc = accuracy(probs, y_train)
            print(f"epoch {epoch + 1:2d}/{epochs}   loss {loss:.4f}   train accuracy {acc:.2%}")

def hidden_layer_sweep(X_train, y_train, X_test, y_test, sizes, epochs, batch_size, learning_rate):
    """
    Train a new network at each hidden layer size and record
    its test accuracy.

    NOTE: this trains a whole network per size so its really slow
    """
    accuracies = []
    for size in sizes:
        # build and train a fresh network of this hidden size
        net = NeuralNetwork(hidden_size=size)
        train(net, X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, verbose=False)

        # measure how well this size did
        test_acc = accuracy(net.forward(X_test), y_test)
        accuracies.append(test_acc)
        print(f"hidden size {size:4d}  ->  test accuracy {test_acc:.2%}")

    return sizes, accuracies

def main():
    # -----------------------
    # define the cli
    parser = argparse.ArgumentParser(description="Train scratchnet on MNIST.")
    parser.add_argument("--epochs", type=int, default=10, help="number of passes over the training data (default=10)")
    parser.add_argument("--batch-size", type=int, default=64, help="images per gradient step (default=64)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--show-lr-graph", action="store_true", help="graph loss vs learning rate before training")
    parser.add_argument("--show-hls-graph", action="store_true", help="train accross multiple hidden layer sizes and graph accuracy vs size (TAKES A MOMENT)")
    parser.add_argument("--show-full-stats", action="store_true", help="print the per digit accuracy and the confusion matrix")
    parser.add_argument("--show-missed-digits", action="store_true", help="draw a few misclassified test digits")
    parser.add_argument("--save", metavar="PATH", help="save the trained model to PATH (.npz)")
    parser.add_argument("--load", metavar="PATH", help="load a trained model from PATH and skip the training")
    parser.add_argument("--shush", action="store_true", help="hide the per epoch training progress")
    args = parser.parse_args()

    # -----------------------
    # load the data and build a fresh network
    X_train, y_train, X_test, y_test = load_mnist()

    # -----------------------
    # hidden layer size experiment: a test to investigate hidden layer size vs accuracy
    if args.show_hls_graph:
        sizes = [16, 32, 64, 128, 256, 512]
        sizes, accuracies = hidden_layer_sweep(X_train, y_train, X_test, y_test, sizes=sizes, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        report.plot_hidden_layer_sweep(sizes, accuracies)
        return

    # -----------------------
    # get a network, either load a trained one, or train a fresh one
    net = NeuralNetwork()
    if args.load:
        # load saved weights and skip training 
        net.load(args.load)
        print(f"loaded model from {args.load}")
    else:
        # optional lr graph first
        if args.show_lr_graph:
            learning_rates, losses = net.lr_range_test(X_train, y_train)
            report.plot_lr_range_test(learning_rates, losses)
        # train the fresh network
        train(net, X_train, y_train,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, verbose=not args.shush)

    # -----------------------
    # the main result, always shown
    test_acc = accuracy(net.forward(X_test), y_test)
    print(f"\ntest accuracy: {test_acc:.2%}")

    # -----------------------
    # optional extra reports
    if args.show_full_stats:
        report.print_full_stats(net, X_test, y_test)
    if args.show_missed_digits:
        report.show_missed_digits(net, X_test, y_test)

    # -----------------------
    # optional: save the trained model
    if args.save:
        net.save(args.save)
        print(f"\nsaved trained model to {args.save}")

if __name__ == "__main__":
    main()

