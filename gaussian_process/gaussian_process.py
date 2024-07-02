import numpy as np

from kernel import Kernel

class GaussianProcess:
    def __init__(self,
                 x1: np.ndarray,
                 y1: np.ndarray,
                 x2: np.ndarray,
                 kernel: Kernel):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.kernel = kernel

        self.mu = 0
        self.sigma = 0

    def fit_and_predict(self, n_predictions: int) -> np.ndarray:
        self.fit()
        return self.predict(n_predictions)

    def fit(self) -> np.ndarray:
        numerical_stability = 1e-7 * np.eye(self.x1.shape[0])

        sigma11 = self.kernel.compute(self.x1, self.x1) + numerical_stability
        sigma22 = self.kernel.compute(self.x2, self.x2)
        sigma12 = self.kernel.compute(self.x1, self.x2)
        sigma21 = sigma12.T

        shared_matrix = np.linalg.solve(sigma11, sigma12).T

        # assume zero mean
        mu2_given_1 = shared_matrix @ self.y1
        sigma2_given_1 = sigma22 - shared_matrix @ sigma12

        self.mu = mu2_given_1
        self.sigma = sigma2_given_1
        return self

    def predict(self, n_predictions: int) -> np.ndarray:
        return np.random.multivariate_normal(mean=self.mu,
                                             cov=self.sigma,
                                             size=n_predictions)
