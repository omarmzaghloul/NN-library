import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, weights, gradients):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        return weights - self.learning_rate * gradients


class Adam(Optimizer):
    def __init__(self):
        pass
    def update(self, weights, gradients):
        pass

