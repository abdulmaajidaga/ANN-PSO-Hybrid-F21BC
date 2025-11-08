import numpy as np


def reconstruct_params(flat_vector, ann):
    """
    Converts a flat 1D particle vector back into the ANN's structured
    parameters (weights, biases, activation list).
    """
    weights, biases, activations = [], [], []
    idx = 0

    layers = ann.layers
    num_layers = ann.num_layers
    activations_functions = ann.activations_functions
    num_activation_functions = ann.num_activation_functions
    num_activation_variables = num_layers - 2
    

    # Extract all weights
    for i in range(num_layers - 1):
        w_size = layers[i] * layers[i+1]
        W = np.array(flat_vector[idx:idx+w_size], dtype=float).reshape(layers[i], layers[i+1])
        idx += w_size
        weights.append(W)

    # Extract all biases
    for i in range(num_layers - 1):
        b_size = layers[i+1]
        b = np.array(flat_vector[idx:idx+b_size], dtype=float).reshape(1, layers[i+1])
        idx += b_size
        biases.append(b)

    # Extract all activation functions
    a_size = num_activation_variables * num_activation_functions
    discrete_variable_probabilities = flat_vector[idx:idx+a_size]
    
    #Select the activation with the highest probability for each variable
    activations = np.array([
        activations_functions[np.argmax(prob_array)]
        for prob_array in discrete_variable_probabilities.reshape(num_activation_variables, num_activation_functions)
    ])
    

    return [weights, biases, activations]