import numpy as np
from scipy.stats import rv_continuous


# First random fourier feature mapping introduced in the paper
def rff_1(X, D):
    """
    :param X: Input data matrix, shape: (N, d)
    :param D: Dimensionality of the randomized feature map
    :return: Randomized fourier feature map matrix Z, shape: (N, 2D). Each row of Z corresponds to a randomized fourier
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
    :return: Randomized fourier feature map matrix Z, shape: (N, D). Each row of Z corresponds to a randomized fourier
    feature map for a data point in X.
    """
    N, d = X.shape
    W = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=D)
    b = np.random.uniform(0, 2 * np.pi, size=D).reshape(1, -1)
    temp = X @ W.T + b
    Z = np.sqrt(2) * np.cos(temp)

    return Z / np.sqrt(D)


# Gamma distribution arises from the laplacian kernel
class Gamma(rv_continuous):
    def _pdf(self, delta):
        return delta * np.exp(-delta)


def bin_index(x, delta, u):
    """
    :param x: Input vector, shape: (d, ). One row of input data matrix X.
    :param delta: Grid pitch parameters, shape: (d, ).
    :param u: Shift parameters, shape: (d, ).
    :return: An index array that represents which bin x lies in, shape: (d, ).
    """
    return np.array([np.ceil((x - u) / delta).astype(int)])


# Random binning feature mapping (Algorithm 2 in the paper)
def rbf(X, P):
    """
    :param X: Input data matrix, shape: (N, d)
    :param P: Number of samples we draw to approximate the kernel.
    :return: Randomized binning feature map matrix Z, shape: (N, \sum_{i=1}^P M_i), where M_i is the number of unique
    bin indices in the ith iteration. Each row of Z corresponds to a randomized binning feature map for a data point in
    X.

    """
    N, d = X.shape
    indices = np.zeros((N, d))  # Matrix that stores bin indices for all training data point
    gamma_dist = Gamma(a=0)  # The lower bound of the support of the distribution is 0, i.e., a, is set to 0
    Z = []

    for p in range(P):
        delta = gamma_dist.rvs(size=d)  # Sample d delta from this distribution
        u = np.random.uniform(0, delta, size=d)

        for i in range(N):
            indices[i, :] = bin_index(X[i, :], delta, u)

        unique_indices = np.unique(indices, axis=0)  # Eliminates unoccupied bins from the representation
        length = unique_indices.shape[0]
        idx_to_pos = {tuple(idx): pos for pos, idx in
                      enumerate(unique_indices)}  # Numpy array, i.e., idx, is not hashable
        one_hot_vectors = np.zeros((N, length), dtype=int)  # Each row corresponds to a z_p(x) for a specific x
        for i in range(N):
            one_hot_vectors[i, idx_to_pos[tuple(indices[i, :])]] = 1

        Z.append(one_hot_vectors)

    return np.hstack(Z) / np.sqrt(P)
