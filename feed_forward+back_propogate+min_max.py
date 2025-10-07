import numpy as np
import pandas as pd

path = "concrete_data.csv"
dataset = pd.read_csv(path)
split_idx = int(dataset.shape[0] * 0.7)
training_dataset = dataset.iloc[:split_idx, :] 
testing_dataset = dataset.iloc[split_idx:, :]

# Min-Max normalization function
def min_max_normalize(data):
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)
    return (data - min_) / (max_ - min_ + 1e-8), min_, max_

training_input_raw = training_dataset.iloc[:, :8].values
training_target_raw = training_dataset.iloc[:, 8].values.reshape(-1, 1)

training_input, input_min, input_max = min_max_normalize(training_input_raw)
training_target, target_min, target_max = min_max_normalize(training_target_raw)

class MultiLayerANN:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            if activations[i] == 'relu':
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
            else:
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i+1])))

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))
    def _d_logistic(self, x):
        sig = self._logistic(x)
        return sig * (1 - sig)
    def _relu(self, x):
        return np.maximum(0, x)
    def _d_relu(self, x):
        return (x > 0).astype(float)
    def _tanh(self, x):
        return np.tanh(x)
    def _d_tanh(self, x):
        return 1 - np.tanh(x) ** 2
    def _linear(self, x):
        return x
    def _d_linear(self, x):
        return np.ones_like(x)

    def _activate(self, x, fn_name):
        match fn_name:
            case 'logistic': return self._logistic(x)
            case 'relu':     return self._relu(x)
            case 'tanh':     return self._tanh(x)
            case 'linear':   return self._linear(x)
            case _:          raise ValueError(f'Unknown activation: {fn_name}')

    def _activate_deriv(self, x, fn_name):
        match fn_name:
            case 'logistic': return self._d_logistic(x)
            case 'relu':     return self._d_relu(x)
            case 'tanh':     return self._d_tanh(x)
            case 'linear':   return self._d_linear(x)
            case _:          raise ValueError(f'Unknown activation: {fn_name}')

    def _forward(self, X):
        self.Zs = []
        self.As = [X]
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z, self.activations[i])
            self.Zs.append(z)
            self.As.append(a)
        return a

    def fit(self, X, y, epochs=5000, lr=0.001):
        for epoch in range(epochs):
            y_pred = self._forward(X)
            loss = np.mean((y_pred - y) ** 2)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Training MSE: {loss:.6f}")
            dA = (y_pred - y) / X.shape[0]
            for i in reversed(range(len(self.weights))):
                dZ = dA * self._activate_deriv(self.Zs[i], self.activations[i])
                dW = np.dot(self.As[i].T, dZ)
                dB = np.sum(dZ, axis=0, keepdims=True)
                dA = np.dot(dZ, self.weights[i].T)
                self.weights[i] -= lr * dW
                self.biases[i]  -= lr * dB

layers = [8, 32, 32, 1]
activations = ['relu', 'relu', 'linear']
ann = MultiLayerANN(layers, activations)
ann.fit(training_input, training_target, epochs=10000, lr=0.001)
