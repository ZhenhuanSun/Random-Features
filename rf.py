import numpy as np
from scipy.stats import rv_continuous
from collections import defaultdict

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


# Gamma distribution arises from laplacian kernel
class Gamma(rv_continuous):
    def _pdf(self, delta):
        return delta * np.exp(-delta)


def bin_index(x, delta, u):
    """
    :param x: Input vector, shape: (d, ). One row of input data matrix X.
    :param delta: Grid pitch parameters, shape: (d, ).
    :param u: Shift parameters, shape: (d, ).
    :return: An index array that represents the bin, shape: (d, ).
    """
    return np.array([np.ceil((x - u) / delta).astype(int)])


# Random binning feature mapping (Algorithm 2 in the paper)
def rbf(X, P):
    """
    :param X: Input data matrix, shape: (N, d)
    :param P: Dimensionality of the randomized feature map
    :return:
    """
    N, d = X.shape
    Z = np.zeros((N, P))
    index_matrix = np.zeros((N, d))
    gamma_dist = Gamma(a=0) # The lower bound of the support of the distribution, i.e., a, is set to 0

    for p in range(P):
        delta = gamma_dist.rvs(size=d) # Sample d delta from this distribution
        u = np.random.uniform(0, delta, size=d)

        for i in range(N):
            index_matrix[i, :] = bin_index(X[i, :], delta, u)

        # Eliminates unoccupied bins from the representation





