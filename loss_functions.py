import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculates the average of the squared differences between predictions and true values."""
    return np.mean((y_pred - y_true) ** 2)

def root_mean_squared_error(y_true, y_pred):
    """Calculates the square root of the mean squared error."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
def mean_absolute_error(y_true, y_pred):
    """Calculates the average of the absolute differences between predictions and true values."""
    return np.mean(np.abs(y_pred - y_true))

# A dictionary to look up loss function objects by their string names
LOSS_FUNCTIONS = {
    'mse': mean_squared_error, 
    'rmse': root_mean_squared_error, 
    'mae': mean_absolute_error
}

def get_loss_function(name):
    """Retrieves a loss function from the registry by its name."""
    func = LOSS_FUNCTIONS.get(name)
    if func is None:
        # Raise an error if the requested loss function name is not found
        raise ValueError(f'Unknown loss function: {name}')
    return func