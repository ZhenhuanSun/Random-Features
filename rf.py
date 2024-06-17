import numpy as np


# First random fourier feature mapping introduced in the paper
def rff_1(X, D):
    """
    :param X: Input data matrix, shape: (N, d)
    :param D: Dimensionality of the randomized feature map
    :return: Scaled randomized fourier feature map matrix Z, shape: (N, 2D). Each row of Z corresponds to a randomized fourier
    feature map for a data point in X.
    """
    d = X.shape[1]
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=D)
    temp = X @ W.T
    cos_features = np.cos(temp)
    sin_features = np.sin(temp)
    Z = np.concatenate((cos_features, sin_features), axis=1)

    return Z / np.sqrt(D)


# Second random fourier feature mapping introduced in the paper
def rff_2(X, D):
    """
    :param X: Input data matrix, shape: (N, d)
    :param D: Dimensionality of the randomized feature map
    :return: Scaled randomized fourier feature map matrix Z, shape: (N, D). Each row of Z corresponds to a randomized fourier
    feature map for a data point in X.
    """
    N, d = X.shape
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=D)
    b = np.random.uniform(0, 2 * np.pi, size=D).reshape(1, -1)
    temp = X @ W.T + b
    Z = np.sqrt(2) * np.cos(temp)

    return Z / np.sqrt(D)
