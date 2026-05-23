import numpy as np

def normalize_data(X):

    X=X/255.0

    print("Data normalized between 0 and 1")

    return X