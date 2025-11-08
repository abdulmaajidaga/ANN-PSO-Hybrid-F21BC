import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
    
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

LOSS_FUNCTIONS = {'mse': mse, 'rmse': rmse, 'mae': mae}

def get_loss_function(name):
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f'Unknown loss function: {name}')
    return LOSS_FUNCTIONS[name]