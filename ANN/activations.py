import numpy as np

# def logistic(x):
#     return np.where(
#         x >= 0,
#         1 / (1 + np.exp(-x)),   # for x >= 0
#         np.exp(x) / (1 + np.exp(x))  # for x < 0
#     )

def logistic(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

ACTIVATIONS = {'logistic': logistic, 'relu': relu, 'tanh': tanh, 'linear': linear}
ACTIVATION_LIST = ['logistic', 'relu', 'tanh', 'linear']

def get_activation_by_name(name):
    if name not in ACTIVATIONS:
        raise ValueError(f'Unknown activation function: {name}')
    return ACTIVATIONS[name]

def get_activation_count():
    return len(ACTIVATION_LIST)

def get_activations():
    return ACTIVATION_LIST

# unused, could be removed if not used by the end of the course
def get_activation_by_index(idx):
    return ACTIVATIONS[ACTIVATION_LIST[idx]]
