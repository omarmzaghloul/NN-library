import numpy as np
from .losses import get_loss_function

class SequentialNN:
    def __init__(self, layers=[]):
        self.layers = layers
        self.optimizer = None
        self.loss_function = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_gradient):
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self):
        for layer in self.layers:
            layer.weights = self.optimizer.update(layer.weights, layer.weights_gradient)
            layer.bias = self.optimizer.update(layer.bias, layer.bias_gradient)
                
    def setting(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = get_loss_function(loss)
      

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=30):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                
                y_pred = self.forward(X_batch)
                
                train_loss = self.loss_function.compute_loss(y_pred, y_batch)
                
                
                loss_gradient = self.loss_function.compute_gradient(y_pred, y_batch)

                self.backward(loss_gradient)
                
                self.update()

            if epoch % 10 == 0:
                val_pred = self.forward(X_val)
                val_loss = self.loss_function.compute_loss(val_pred, y_val)
                pred = self.forward(X_train)
                train_loss = self.loss_function.compute_loss(pred, y_train)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")

    def predict(self, X):
        return self.forward(X)
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_pred = (y_pred > 0.5).astype(int)
        return np.mean(y_pred == y_true)
                
