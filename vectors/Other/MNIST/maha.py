import numpy as np

def mahalanobis(u, v, VI):
    VI = np.atleast_2d(VI)
    delta = np.subtract(u, v)
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)

def cov_matrix(data):
    return np.cov(data.T)

def cov_standard(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    matrix = (matrix - mean) / std
    return np.cov(matrix.T)