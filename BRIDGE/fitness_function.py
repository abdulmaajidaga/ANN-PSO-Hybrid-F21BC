import numpy as np
import ANN.loss_functions as loss_functions
from .particle_decoder import reconstruct_params


def create_objective_function(ann, X_train, y_train, loss_function_name='mse'): 
    """
    A "factory" that returns the objective function the PSO will minimize.
    This function "closes over" the ann, X_train, and y_train variables
    so they are available to the inner function.
    """
    
    # Get the actual loss function from the registry
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
            loss = loss_func(y_train, predictions) 
            loss_array.append(loss)
            
        return np.array(loss_array)

    # Return the *function itself*, not the result of calling it.
    return objective_function