import numpy as np
import ANN.activations as act

class MultiLayerANN:
    
    def __init__(self, layers, activations = None):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        self.activations_functions = act.get_activations()
        self.num_activation_functions = act.get_activation_count()
        

        for i in range(self.num_layers - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)  # 0.1 is the scaling factor
            self.biases.append(np.zeros((1, layers[i+1]))) # Bias is initialized to zero
    
        if activations is not None:
            self.activations = activations
        else:
            self.activations = ['relu'] * (len(layers) - 2)

    def compute_forward(self, X, weights, biases, activation_names):
        a = X  # X is the input array
        num_hidden_layers = len(weights) - 1 

        act_funcs = [act.get_activation_by_name(name) for name in activation_names]

        for i in range(num_hidden_layers):
            z = np.dot(a, weights[i]) + biases[i]
            a = act_funcs[i](z)

        z = np.dot(a, weights[num_hidden_layers]) + biases[num_hidden_layers]    
        return z

    # Convenience wrapper
    def predict(self, X):
        return self.compute_forward(X, self.weights, self.biases, self.activations)

    # added by Dilon
    def evaluate_with_params(self, X, params):
        weights = params[0]
        biases = params[1] 
        if len(params) > 2:
            activations = params[2]
        else: 
            activations = self.activations
        
        return self.compute_forward(X, weights, biases, activations)