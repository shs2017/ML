import numpy as np

from abc import ABC, abstractmethod


class Kernel(ABC):
    @abstractmethod
    def compute(x1: np.ndarray, x2: np.ndarray):
        raise NotImplementedError


class RBFKernel(Kernel):
    def __init__(self, sigma: float):
        super().__init__()

        self.sigma = sigma

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1 = np.expand_dims(x1, -1)
        x2 = np.expand_dims(x2, -1).T

        return np.exp((-1 / (2 * self.sigma**2)) * np.abs(x1 - x2)**2)
