import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def forward(self, z):
        pass

    @abstractmethod
    def backward(self, loss):
        pass
    
class ReLU(Activation):
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, loss):
        return loss * (self.z > 0)

class Sigmoid(Activation):
    def forward(self, z):
        self.z = z
        return 1 / (1 + np.exp(-z))

    def backward(self, loss):
        sig = 1 / (1 + np.exp(-self.z))
        return loss * sig * (1 - sig)
    
class Linear(Activation):
    def forward(self, z):
        self.z = z
        return z  

    def backward(self, loss):
        return loss    
    
def get_activation(name):
    activations = {
        "relu": ReLU(),
        "sigmoid": Sigmoid(),
        "linear": Linear()
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation function: {name}")
    return activations[name.lower()]    