import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    np_array = np.array(x)
    val = 1 / (1 + np.exp(-np_array))

    return val