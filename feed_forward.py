import numpy as np
import pandas as pd

path = "concrete_data.csv"
dataset = pd.read_csv(path)

split_idx = int(dataset.shape[0] * 0.7) # iloc takes int and not floats
training_dataset = dataset.iloc[:split_idx, :] 
testing_dataset = dataset.iloc[split_idx:, :]

training_input = training_dataset.iloc[:, :8].values

class MultiLayerANN:
    def __init__(self, layers, activations, params = None):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []

        if params is not None:
            self.weights = params[0]
            self.biases = params[1]
        else:
            # --- Create random weights & biases for each layer ---
            for i in range(len(layers) - 1):
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
                self.biases.append(np.zeros((1, layers[i+1])))
       
        
        
    
    
    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def _activate(self, x, fn_name):
        match fn_name:
            case 'logistic':
                return self._logistic(x)
            case 'relu':
                return self._relu(x)
            case 'tanh':
                return self._tanh(x)
            case 'linear':
                return x
            case _:
                raise ValueError(f'Unknown activation function: {fn_name}')

    def _forward(self, X):
        
        num_hidden_layers = len(self.layers) - 2
        a = X  
        for i in range(num_hidden_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i] 
            a = self._activate(z, self.activations[i])
        
        z = np.dot(a, self.weights[num_hidden_layers]) + self.biases[num_hidden_layers]
        return z



# layers = [8, 2, 2, 1]
# activations = ['relu', 'relu']
# predictions = MultiLayerANN(layers, activations)._forward(training_input)
# print(predictions)
    