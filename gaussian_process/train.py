import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from gaussian_process import GaussianProcess
from kernel import RBFKernel

if __name__ == '__main__':
    # Create Data
    x1_min = -10
    x1_max = 10

    x2_min = -12
    x2_max = 12

    x1 = np.random.uniform(x1_min, x1_max, size=100)
    y1 = x1**2 + np.random.randn(x1.shape[0]) * 0.1
    x2 = np.linspace(x2_min, x2_max, 50)

    # Create and fit model
    rbf_kernel = RBFKernel(sigma=5)
    gp = GaussianProcess(x1, y1, x2, rbf_kernel)
    y2_predictions = gp.fit_and_predict(n_predictions=100)

    # Plot
    plt.scatter(x1, y1, c='blue', marker='x', s=20)
    for y2 in y2_predictions:
        plt.plot(x2, y2, linewidth=0.5)
    plt.show()
