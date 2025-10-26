import numpy as np
import loss_functions

def initialize_particles(ann, num_particles):
    """
    Creates an initial population of particles for PSO.
    Each particle is a flat 1D vector encoding all ANN parameters 
    (weights, biases, and activation function indices).
    """
    particles = []

    num_layers = ann.num_layers
    # Number of learnable activation functions (one for each hidden layer)
    num_activations_variables = num_layers - 2 
    num_activation_functions = ann.num_activation_functions
    
    for _ in range(num_particles):
        weights = []
        biases = []
        # Initialize weights and biases for all layers
        for i in range(num_layers - 1):
            # Small random weights
            weights.append(np.random.randn(ann.layers[i], ann.layers[i+1]) * 0.1) 
            # Zero biases
            biases.append(np.zeros((1, ann.layers[i+1])))
        
        # Initialize activation functions with random integer indices
        activations = np.random.randint(0, num_activation_functions, 
                                        size=(num_activations_variables,))
        
        # Flatten all parameters into a single 1D vector
        flat_vector = np.concatenate(
            [W.flatten() for W in weights] +
            [b.flatten() for b in biases] +
            [activations.flatten()]
        )
        particles.append(flat_vector)
        
    return np.array(particles)

def reconstruct_params(flat_vector, ann):
    """
    Converts a flat 1D particle vector back into the ANN's structured
    parameters (weights, biases, activation list).
    """
    weights, biases = [], []
    idx = 0 # Current position in the flat vector
    layers = ann.layers
    num_layers = ann.num_layers
    num_activations_variables = num_layers - 2

    # Reconstruct weight matrices
    for i in range(num_layers - 1):
        w_size = layers[i] * layers[i+1]
        W = flat_vector[idx:idx+w_size].reshape(layers[i], layers[i+1])
        idx += w_size
        weights.append(W)

    # Reconstruct bias vectors
    for i in range(num_layers - 1):
        b_size = layers[i+1]
        b = flat_vector[idx:idx+b_size].reshape(1, layers[i+1])
        idx += b_size
        biases.append(b)

    # Reconstruct activation function indices
    a_size = num_activations_variables
    activations_float = flat_vector[idx:idx+a_size]
    max_index = ann.num_activation_functions - 1
    
    # Clip values to be within the valid index range [0, max_index]
    # This is crucial as PSO operates in continuous space
    activations_clipped = np.clip(activations_float, 0, max_index)
    
    # Convert to integers to be used as indices
    activations = activations_clipped.astype(int).tolist()

    return [weights, biases, activations]

# 2. ADD NEW PARAM
def create_objective_function(ann, X_train, y_train, loss_function_name='mse'): 
    """
    A "factory" that returns the objective function the PSO will minimize.
    This function "closes over" the ann, X_train, and y_train variables
    so they are available to the inner function.
    """
    
    # 3. Get the actual loss function from the registry
    loss_func = loss_functions.get_loss_function(loss_function_name)
    
    def objective_function(particles):
        """
        Calculates the fitness (loss) for a batch of particles.
        This is the function that PSO will call repeatedly.
        """
        loss_array = []
        for particle in particles:
            # 1. Reconstruct the ANN parameters from the particle's 1D vector
            params = reconstruct_params(particle, ann)
            
            # 2. Get model predictions using these parameters
            predictions = ann.evaluate_with_params(X_train, params=params)
            
            # 3. Calculate error (fitness) using the chosen loss function
            # 4. USE THE CHOSEN FUNC
            loss = loss_func(y_train, predictions) 
            loss_array.append(loss)
            
        return np.array(loss_array)

    # Return the *function itself*, not the result of calling it.
    return objective_function