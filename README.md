# Neural Network Library

A lightweight, from-scratch implementation of a neural network library in Python. This project provides a simple and educational framework for building, training, and using neural networks with support for various activation functions, loss functions, and optimizers.

## Features

- **Sequential Neural Network Model** - Build neural networks layer by layer
- **Dense Layers** - Fully connected layers with customizable activation functions
- **Activation Functions**
  - ReLU (Rectified Linear Unit)
  - Sigmoid
  - Linear
- **Loss Functions**
  - Mean Squared Error (MSE)
  - Binary Cross Entropy
- **Optimizers**
  - Stochastic Gradient Descent (SGD)
- **Backpropagation** - Automatic gradient computation and weight updates
- **Batch Training** - Support for mini-batch gradient descent
- **Validation** - Training with validation set evaluation

## Project Structure

```
NeuralNetwork/
├── __init__.py          # Package initialization
├── model.py             # Sequential neural network model
├── layers.py            # Dense layer implementation
├── activations.py       # Activation functions
├── losses.py            # Loss functions
├── optimizers.py        # Optimization algorithms
```

## Installation

### Requirements

- Python 3.7+
- NumPy

### Setup

1. Clone or download the project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from NeuralNetwork.model import SequentialNN
from NeuralNetwork.layers import DenseLayer
from NeuralNetwork.optimizers import SGD
import numpy as np

# Create a sequential model
model = SequentialNN()

# Add layers
model.add(DenseLayer(input_size=2, neurons=4, activation='relu'))
model.add(DenseLayer(input_size=4, neurons=1, activation='sigmoid'))

# Configure model with optimizer and loss
model.setting(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy')

# Prepare training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])
X_val = X_train
y_val = y_train

# Train the model
model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=2)

# Make predictions
predictions = model.predict(X_train)
print(predictions)
```

### Available Activation Functions

- `'relu'` - Rectified Linear Unit
- `'sigmoid'` - Sigmoid activation
- `'linear'` - Linear activation (identity)

### Available Loss Functions

- `'mse'` - Mean Squared Error
- `'binary_crossentropy'` - Binary Cross Entropy

### Available Optimizers

- `SGD(learning_rate=0.01)` - Stochastic Gradient Descent with configurable learning rate

## Model Architecture

### SequentialNN

The main model class that builds a neural network as a sequence of layers.

**Methods:**
- `add(layer)` - Add a layer to the network
- `setting(optimizer, loss)` - Configure optimizer and loss function
- `train(X_train, y_train, X_val, y_val, epochs, batch_size)` - Train the model
- `predict(X)` - Make predictions on input data
- `forward(x)` - Forward pass through all layers
- `backward(loss_gradient)` - Backward pass (backpropagation)

### DenseLayer

A fully connected (dense) layer.

**Parameters:**
- `input_size` - Number of input features
- `neurons` - Number of neurons/output units
- `activation` - Activation function name

## Training Process

1. **Forward Pass** - Compute predictions through all layers
2. **Loss Computation** - Calculate the loss between predictions and targets
3. **Backward Pass** - Compute gradients through backpropagation
4. **Weight Update** - Update weights and biases using the optimizer
5. **Validation** - Evaluate on validation set every 10 epochs

