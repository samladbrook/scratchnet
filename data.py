import gzip
import os
import urllib.request

import numpy as np

# get the data
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def download(filename, data_dir):
    """Download one file into data_dir if its not already there"""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def read_images(path):
    """Read an IDX image file into a (num_images, 784) array"""
    with gzip.open(path, "rb") as file:
        file.read(4) # skip the magic number
        num = int.from_bytes(file.read(4), "big") # how many images
        rows = int.from_bytes(file.read(4), "big") # 28
        cols = int.from_bytes(file.read(4), "big") # 28
        buf = file.read(num * rows * cols) # all the pixels
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num, rows * cols)

def read_labels(path):
    """Read an IDX label file into a (num_labels,) array"""
    with gzip.open(path, "rb") as file:
        file.read(4)
        num = int.from_bytes(file.read(4), "big")
        buf = file.read(num)
        return np.frombuffer(buf, dtype=np.uint8)

def load_mnist(data_dir="data"):
    """Donload and return mnist as numpy array

    Returns:
        X_train: (60000, 784) in [0, 1]
        y_train: (60000) in 0..9
        X_test: (10000, 784) in [0, 1]
        y_test: (10000) in 0..9
    """
    paths = {k: download(v, data_dir) for k, v in FILES.items()}
    X_train = read_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = read_labels(paths["train_labels"])
    X_test = read_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test = read_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test