import rf
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_comparisons(X, kernel, kernel_name, feature_map, feature_map_name):
    D_values = [1, 10, 100, 1000]
    colors = ['red', 'green', 'blue', 'purple']
    markers = ['o', 's', '^', 'd']

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 1 + len(D_values), height_ratios=[1, 1], figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(X, kernel[50, :], label=f'Exact {kernel_name} Kernel', color='black', linewidth=2)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(kernel)
    ax2.set_title(f'Exact {kernel_name} Kernel')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    for i, D in enumerate(D_values):
        ax = fig.add_subplot(gs[1, i + 1])
        Z = feature_map(X, D)
        kernel_approximate = Z @ Z.T
        ax.imshow(kernel_approximate)
        ax.set_title(f'{feature_map_name} approximation (D={D})')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax1.plot(X, kernel_approximate[50, :], label=f'{feature_map_name} approximation (D={D})', color=colors[i],
                 linestyle='--', marker=markers[i], markevery=5)

    ax1.set_title(f'{feature_map_name} approximation of {kernel_name} Kernel (1D)', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Kernel value')
    ax1.legend(loc='best')
    ax1.grid(True)

    plt.tight_layout()


if __name__ == '__main__':
    x = np.linspace(-5, 5, 101).reshape(-1, 1)
    gaussian_kernel = rbf_kernel(x, None, 1/2)  # Gaussian kernel with \sigma = 1
    laplacian_kernel = laplacian_kernel(x, None, 1)  # Laplacian kernel with gamma = 1

    plot_comparisons(x, gaussian_kernel, 'Gaussian', rf.rff_1, 'rff_1')
    plot_comparisons(x, gaussian_kernel, 'Gaussian', rf.rff_2, 'rff_2')
    plot_comparisons(x, laplacian_kernel, 'Laplacian', rf.rbf, 'rbf')

    plt.show()
