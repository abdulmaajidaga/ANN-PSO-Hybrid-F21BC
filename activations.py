import numpy as np

def logistic(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def linear(x):
    return x

# A dictionary to look up function objects by their string names
ACTIVATION_FUNCTIONS = {'logistic':logistic,'relu':relu,'tanh':tanh,'linear':linear}

# A list to map integer indices (e.g., from an optimization algorithm like PSO) 
# to their corresponding string names
ACTIVATION_MAP = ['logistic', 'relu', 'tanh', 'linear']

def get_activation_by_name(fn_name):
    func = ACTIVATION_FUNCTIONS.get(fn_name)
    if func is None:
        raise ValueError(f'Unknown activation function: {fn_name}')
    return func

def get_name_by_index(index):
    try:
        return ACTIVATION_MAP[index]
    except IndexError:
        raise ValueError(f'Unknown activation function index: {index}')

def get_activation_count():
    return len(ACTIVATION_MAP)