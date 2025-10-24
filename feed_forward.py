import numpy as np
import pandas as pd

path = "concrete_data.csv"
dataset = pd.read_csv(path)

split_idx = int(dataset.shape[0] * 0.7) # iloc takes int and not floats
training_dataset = dataset.iloc[:split_idx, :] 
testing_dataset = dataset.iloc[split_idx:, :]

training_input = training_dataset.iloc[:, :8].values

class MultiLayerANN:
    # # ABDUL 
    # Add loss function paramater and functions
    def __init__(self, layers, activations = None):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self.activation_options = ["logistic", "relu", "tanh", "linear"]

        # --- Create random weights & biases for each layer ---
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i+1])))
              
        if activations is not None:
            self.activations = activations
        else:
            self.activations = ['relu'] * (len(layers) - 2) # Default activations
            


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

    def _match_activations(self, activations_index):
        activations = []
        for index in activations_index:
            match index:
                case 0:
                    name = 'logistic'
                case 1: 
                    name = 'relu'
                case 2:
                    name = 'tanh'
                case 3:
                    name = 'linear'
                case _:
                    raise ValueError(f'Unknown activation function index: {index}')
            activations.append(name)
        return activations


    def _forward(self, X, params = None):
        
        
        if params is not None:
            # Run feedforward with the custom params
            weights = params[0]
            biases = params[1]
            activations = params[2]
        else:
            # Use existing params in the ANN class
            weights = self.weights
            biases = self.biases
            activations = self.activations
            
        num_hidden_layers = self.num_layers - 2
        a = X  
        for i in range(num_hidden_layers):
            z = np.dot(a, weights[i]) + biases[i]
            a = self._activate(z, activations[i])

        z = np.dot(a, weights[num_hidden_layers]) + biases[num_hidden_layers]
        
        return z
    