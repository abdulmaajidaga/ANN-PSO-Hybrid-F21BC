import numpy as np
import activations as act # Import the previously defined activation functions

class MultiLayerANN:
    """
    Represents a Multi-Layer Artificial Neural Network (ANN).
    """
    
    def __init__(self, layers, activations = None):
        """
        Initializes the neural network structure, weights, and biases.
        
        Args:
            layers (list): A list of integers specifying the number of neurons in each
                           layer, e.g., [input_size, hidden1_size, output_size].
            activations (list, optional): A list of string names for the activation
                                          functions of the hidden layers. Defaults to 'relu'.
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = []
        self.biases = []
        # Get the total number of available activation functions from the 'activations' module
        self.num_activation_functions = act.get_activation_count()

        # Initialize weights and biases for connections between layers
        for i in range(self.num_layers - 1):
            # Small random weights
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1) 
            # Zero biases
            self.biases.append(np.zeros((1, layers[i+1])))
              
        if activations is not None:
            self.activations = activations
        else:
            # Default to ReLU for all hidden layers if none are specified
            # The -2 accounts for the input and output layers
            self.activations = ['relu'] * (len(layers) - 2)

    @staticmethod
    def compute_forward(X, weights, biases, activation_names):
        """
        Performs a forward pass through the network using a given set of parameters.
        This is a static method, meaning it doesn't rely on the instance's (self)
        own weights or biases, making it usable by external processes like PSO.
        """
        a = X # 'a' represents the activation from the previous layer, starting with input X
        num_hidden_layers = len(weights) - 1 

        # Get the actual function objects from their string names
        act_funcs = [act.get_activation_by_name(name) for name in activation_names]

        # Process all hidden layers
        for i in range(num_hidden_layers):
            z = np.dot(a, weights[i]) + biases[i] # Linear combination
            a = act_funcs[i](z)                   # Apply activation function

        # Process the output layer (no activation function applied here, it's linear)
        z = np.dot(a, weights[num_hidden_layers]) + biases[num_hidden_layers]    
        return z

    def predict(self, X):
        """
        Performs a forward pass using the network's *own* stored parameters.
        """
        return self.compute_forward(X, self.weights, self.biases, self.activations)

    def evaluate_with_params(self, X, params):
        """
        A helper method for the PSO. It performs a forward pass using parameters
        provided externally (e.g., from a PSO particle).
        
        Args:
            X: The input data.
            params (list): A list [weights, biases, activation_indices]
        """
        weights = params[0]
        biases = params[1]
        activation_indices = params[2]
        # Convert integer indices (from PSO) to string names
        activation_names = [act.get_name_by_index(i) for i in activation_indices]
        
        # Use the static forward pass method with the provided parameters
        return self.compute_forward(X, weights, biases, activation_names)