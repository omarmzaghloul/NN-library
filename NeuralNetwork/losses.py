import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    @abstractmethod
    def compute_gradient(self, y_pred, y_true):
        pass

class MeanSquaredError(Loss):
    def compute_loss(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def compute_gradient(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class BinaryCrossEntropy(Loss):
    def compute_loss(self, y_pred, y_true):
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_gradient(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

        
def get_loss_function(name):
    loss_functions = {
        "binary_cross_entropy": BinaryCrossEntropy(),
        "mean_squared_error": MeanSquaredError()
    }
    if name.lower() not in loss_functions:
        raise ValueError(f"Unsupported loss function: {name}")
    return loss_functions[name.lower()]