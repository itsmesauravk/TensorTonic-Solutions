import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, D = X.shape # N samples, D features
    w = np.zeros(D)
    b = 0.0

    for step in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        dz = p - y
        grad_w = (1/N) * np.dot(X.T, dz)
        grad_b = (1/N) * np.sum(dz)


        w -= lr * grad_w
        b -= lr * grad_b
        

    return w, b
