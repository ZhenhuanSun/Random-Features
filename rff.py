import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot as plt


x = np.linspace(-5, 5, 100).reshape(-1, 1)
gaussian_kernel_exact = rbf_kernel(x, None, 1/2)

plt.figure(figsize=(14, 8))

plt.plot(x.flatten(), gaussian_kernel_exact[50, :], label='Exact Gaussian', color='black', linewidth=2)

plt.xlabel('Input Value', fontsize=12)
plt.ylabel('Kernel Value', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()