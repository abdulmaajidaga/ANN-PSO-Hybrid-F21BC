"""
Bridge module for integrating ANN with PSO optimization.
Handles encoding/decoding of neural network parameters to/from particle representation.
"""

from .particle_encoder import initialize_particles
from .particle_decoder import reconstruct_params
from .fitness_function import create_objective_function

__all__ = [
    'initialize_particles',
    'reconstruct_params',
    'create_objective_function'
]