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

def suggest_lr(learning_rates, losses):
    """
    Pick a sensible learning rate from a range test. This is the point where
    the loss is dropping the fastest on the way to minimun. that steepest point
    is the quickest rate still improving
    """
    lrs = np.array(learning_rates)
    loss = np.array(losses)

    # smooth the noisy losses with a moving average
    # this is needed otherwise a spike would look like
    # a steep drop
    window = 7
    kernel = np.ones(window) / window
    smooth = np.convolve(loss, kernel, mode="same")

    # the loss bottoms out at this point and we only need the decent up to here
    bottom = int(np.argmin(smooth))

    # slopw of the losses vs log of lr, the biggest negative is the steepest drop
    slopes = np.gradient(smooth, np.log(lrs))
    start = min(window, bottom)
    if bottom <= start:
        return lrs[bottom], loss[bottom]
    best = start + int(np.argmin(slopes[start:bottom + 1]))
    return lrs[best], loss[best]

def plot_lr_range_test(learning_rates, losses, save_path=None):
    """
    Graph loss vs learning rate from the lr range test method
    """
    # get the suggested rate
    ideal_lr, ideal_loss = suggest_lr(learning_rates, losses)

    # draw the curve
    plt.plot(learning_rates, losses)
    # need to log scale to spread exponential out
    plt.xscale("log")
    # a dot and dashed line showing ideal lr
    plt.axvline(ideal_lr, color="red", linestyle="--", alpha=0.6)
    plt.scatter([ideal_lr], [ideal_loss], color="red", zorder=5)
    plt.annotate(f"LR = {ideal_lr:.3f}\nLoss = {ideal_loss:.3f}", (ideal_lr, ideal_loss), xytext=(10, 80), textcoords="offset points")
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("scratchnet learning rate range test")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"saved graph to {save_path}")
    else:
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

def plot_hidden_layer_sweep(sizes, accuracies, human_accuracy=99.8, save_path=None):
    """
    Graph the test accuracy against the hidden layer size
    with a reference line at the approx human performance
    """
    percents = [a * 100 for a in accuracies]
    # marker="o" puts a dot on each size
    plt.plot(sizes, percents, marker="o", label="scratchnet")
    plt.xscale("log", base=2)
    # horizontal line that is human avg
    plt.axhline(human_accuracy, color="green", linestyle="--", label=f"approx human (~{human_accuracy:.1f}%)")
    plt.xlabel("hidden layer size (neurons)")
    plt.ylabel("test accuracy (%)")
    plt.title("scratchnet accuracy vs hidden layer size")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"saved graph to {save_path}")
    else:
        plt.show()
