from NeuralNetwork.layers import DenseLayer
from NeuralNetwork.model import SequentialNN
from NeuralNetwork.optimizers import SGD


model = SequentialNN([
    DenseLayer(input_size=X_train.shape[1], neurons=64, activation='relu'),
    DenseLayer(input_size=64 ,neurons=32, activation='relu'),
    DenseLayer(input_size=32 ,neurons=8, activation='relu'),
    DenseLayer(input_size=8 ,neurons=1, activation='sigmoid')
])

model.setting(optimizer=SGD(), loss='binary_crossen_tropy')

model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)

