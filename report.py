import numpy as np
from network import confusion_matrix
import matplotlib.pyplot as plt


def render_digit(image):
    """
    Print one MNIST image to the terminal
    """
    # AI generated way to visualise images in terminal
    shades = " .:-=+*#%@"
    grid = image.reshape(28, 28)
    for row in grid:
        line = ""
        for pixel in row:
            index = int(pixel * (len(shades)-1))
            line += shades[index] *2
        print(line)


def plot_lr_range_test(learning_rates, losses):
    """
    Graph loss vs learning rate from the lr range test method
    """
    # draw the curve
    plt.plot(learning_rates, losses)
    # need to log scale to spread exponential out
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("scrathnet learning rate range test")
    plt.grid(True)
    plt.show()


def print_full_stats(net, X_test, y_test):
	"""
	Print per digit accuracy and the confusion matrix
	"""
	# -----------------------
	# evaluation on the held out test set
	probs = net.forward(X_test)
	predictions = probs.argmax(axis=1)

	# -----------------------
	# per digit accuracy
	print(f"\naccuracy per digit:")
	for digit in range(10):
	    # get the test images that labels are the digit
	    mask = (y_test == digit)
	    digit_acc = np.mean(predictions[mask] == digit)
	    print(f"    {digit}:    {digit_acc:.2%} ({mask.sum()} images)")

	# -----------------------
	# confusion matrix
	cm = confusion_matrix(y_test, predictions)
	print("\nconfusion matrix (row = actual digit, column = predicted)")
	print("      " + "".join(f"{d:5d}" for d in range(10)))
	for true_digit in range(10):
	    row = "".join(f"{count:5d}" for count in cm[true_digit])
	    print(f"  {true_digit}: {row}")


def show_missed_digits(net, X_test, y_test, count=3):
	"""
	Render some images that were predicted wrong
	"""
	predictions = net.predict(X_test)
	# find all positions where the guess didnt match
	wrong = np.where(predictions != y_test)[0]
	print(f"3 of the {len(wrong)} mistakes that were made:\n")
	for i in wrong[:3]:
	    print(f"true digit: {y_test[i]}     network guessed: {predictions[i]}")
	    render_digit(X_test[i])
	    print()
