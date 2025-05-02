import numpy as np
from .activations import get_activation


class DenseLayer:
    def __init__(self, input_size, neurons, activation):
        self.input_size = input_size
        self.neurons = neurons
        self.activation = get_activation(activation)
        
        self.weights = np.random.randn(input_size, neurons) * 0.01
        self.bias = np.zeros((1, neurons))
        
        self.input = None
        self.linear_output = None
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, input):
        self.input = input
        self.linear_output = np.dot(input, self.weights) + self.bias
        return self.activation.forward(self.linear_output)

    def backward(self, grad_output):
        activation_gradient = self.activation.backward(grad_output)
        
        self.weights_gradient = np.dot(self.input.T, activation_gradient)
        self.bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        return input_gradient

    
