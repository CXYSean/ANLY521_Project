import pandas as pd
import numpy as np
import os
import glob


def import_data(filepath):
    folder_path = filepath
    training = []
    authors = []
    for filename in glob.glob(os.path.join(folder_path, "*/")):
        author = filename.split('/')[-2]
        for article in glob.glob(os.path.join(filename, '*.txt')):
            with open(article, 'r') as file:
                data = file.read()
                training.append(data)
                authors.append(author)

    return training, authors


def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


def shuffle(X, y):
    np.random.seed(521)
    new_order = np.random.permutation(len(X))

    X_shuffle = np.asarray(X)
    y_shuffle = np.asarray(y)
    return X_shuffle[new_order], y_shuffle[new_order]

