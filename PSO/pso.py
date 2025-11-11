# pso.py
import numpy as np


class ParticleSwarm(object):
    def  __init__(self, num_particles, num_informants,  objective_function,  particle_length, discrete_params = None,
                 alpha = 0.729, beta = 1.494, gamma = 1.494, delta = 1.494, epsilon = 0.85, particles = None, v_max_scale = 0.5):
        
        self.num_particles = num_particles # Number of particles in the swarm (swarm size)
        self.objective_function = objective_function # Objective function to be minimized
        self.particle_length = particle_length # Total particle length
        self.discrete_params = discrete_params
    
        self.alpha = alpha # α (alpha): proportion of previous velocity to be retained 
        self.beta = beta # β (beta): proportion of personal best to be retained 
        self.gamma = gamma # γ (gamma): proportion of informants' best to be retained 
        self.delta = delta # δ (delta): proportion of global best to be retained 
        self.epsilon = epsilon # ε (epsilon): jump size or step size of a particle 
        
        # Visualization variables created by chatGPT
        self.mean_fitness = 0
        self.std_fitness = 0
        self.gbest_value_history = []
        self.mean_fitness_history = []
        self.std_fitness_history = []
        self.particle_history = []
        self.velocity_history = []
        self.fitness_history = []
        self.gbest_position_history = []
        
        if particles is not None:
            self.particle_array = particles # Initialise with existing particles 
        else:
            self.particle_array = np.random.uniform(size=(num_particles, self.particle_length)) # Initialise with random particle array

        
        # Reference -> Improved Particle Swarm Optimization Based on Velocity Clamping and Particle Penalization
        # Set particle and velocity limits
        max = self.particle_array.max()
        min = self.particle_array.min()
        # Set particle limits to a high upper and lower bounds based on the original initialization
        p_max_scale = 10
        self.p_max = np.full_like(self.particle_array, max * p_max_scale)
        self.p_min = np.full_like(self.particle_array, min * p_max_scale)
        # Set velocity limit to a scaled value based on the original initialization
        self.v_max = (max - min) * p_max_scale * v_max_scale
        print(self.v_max)
        self.v_min = -self.v_max
        
        ## Here, the particle velocities are randomly initialized in a new array, where each velocity array index matches the particle array index
        ## For example, particle_array[0] has a velocity of velocity_array[0]
        self.velocity_array = np.random.uniform(size=(num_particles, self.particle_length))

        # Stores personal best positions for each particle in the matching index
        # For example, particle_pbest[0] stores the particle pbest for particle_array[0]
        self.particle_pbest = np.copy(self.particle_array)
        
        # Stores fitness values for each particle in the matching index
        self.fitness_values_array =  self.objective_function(self.particle_array) 
        self.personal_best_fitness_values = np.copy(self.fitness_values_array) 
        
        # Store informant best positions for each particle, where each informant position matches the particle array index
        # For example,  particle_ibest[0] stores the particle Ibest for particle_array[0]
        # and particle_ibest_list[0] contains the list of selected informants for particle_array[0] 
        self.particle_ibest, self.particle_ibest_list = self.create_particle_informants(num_informants)
        
        
        # Find and store global best position and value
        global_best_idx = np.argmin(self.fitness_values_array)
        self.gbest = self.particle_array[global_best_idx] 
        self.gbest_value = self.fitness_values_array[global_best_idx] 
        
        # --- Variables for visualization --- genereated by chatgpt
        mean_fitness = np.mean(self.fitness_values_array)
        self.mean_fitness_history.append(mean_fitness)
        self.gbest_value_history.append(self.gbest_value)
        self.particle_history.append(self.particle_array.copy())
        self.velocity_history.append(self.velocity_array.copy())
        self.fitness_history.append(self.fitness_values_array.copy())
        self.gbest_position_history.append(self.gbest.copy())      

    def create_particle_informants(self, num_informants):
        num_particles = self.num_particles
        particle_array = self.particle_array
        particle_ibest = np.zeros_like(particle_array)
        particle_ibest_list = np.zeros((num_particles, num_informants), dtype=int)

        # --- Compute pairwise squared Euclidean distances ---
        # distance_matrix[i,j] = ||particle[i] - particle[j]||^2
        diff = particle_array[:, np.newaxis, :] - particle_array[np.newaxis, :, :]  # shape (N,N,D)
        distance_matrix = np.sum(diff ** 2, axis=2)  # shape (N, N)
        
        
        # --- Loop over each particle to select nearest informants ---
        for i in range(num_particles):
            distances = distance_matrix[i]
            distances[i] = np.inf  # exclude self

            # Indices of closest particles
            nearest_indices = np.argsort(distances)[:num_informants]
            particle_ibest_list[i] = nearest_indices
            # Get the *personal best fitness* of these informants
            informant_pbest_values = self.personal_best_fitness_values[nearest_indices]
            # Find the best one *among the informants*
            best_local_idx = np.argmin(informant_pbest_values)
            # Get the global index of that best informant
            best_global_idx = nearest_indices[best_local_idx]
            # Store the *position* of that best informant's pbest
            particle_ibest[i] = self.particle_pbest[best_global_idx]
            

        return particle_ibest, particle_ibest_list
      
    def _update_informants_best(self):
        """
        Calculates the best informant for each particle *for this iteration*.
        This uses the *personal best* positions of the informants as the
        source of social knowledge.
        """
        for i in range(self.num_particles):
            # Get the *personal best fitness* of these informants
            informants_list = self.particle_ibest_list[i]
            informant_pbest_values = self.personal_best_fitness_values[informants_list]
            # Find the best one *among the informants*
            best_local_idx = np.argmin(informant_pbest_values)
            # Get the global index of that best informant
            best_global_idx = informants_list[best_local_idx]
            # Store the *position* of that best informant's pbest
            self.particle_ibest[i] = self.particle_pbest[best_global_idx]
            
    def _normalize_discrete_variables(self):
        ## Generated by ChatGPT 
        # Normalize discrete variables in each particle to maintain probability distributions for each discrete variable
        num_discrete_variables = self.discrete_params[0]      # discrete variable count
        num_discrete_options = self.discrete_params[1]      # options per discrete variable
        num_continuous_variables = self.particle_length - (num_discrete_variables*num_discrete_options)
        # Extract discrete section and reshape
        discrete_section = self.particle_array[:, num_continuous_variables:].reshape(self.num_particles, num_discrete_variables, num_discrete_options)
        # Normalize each probability distribution along the last axis
        discrete_sum = discrete_section.sum(axis=2, keepdims=True)
        discrete_sum[discrete_sum == 0] = 1.0   # avoid division by zero
        discrete_section /= discrete_sum
        # Write it back inline
        self.particle_array[:, num_continuous_variables:] = discrete_section.reshape(self.num_particles, -1)
           
    def _update(self):
    
        cognitive_component_array =  np.random.uniform(size=(self.num_particles, self.particle_length))
        local_social_component_array = np.random.uniform( size=(self.num_particles, self.particle_length))
        global_social_component_array = np.random.uniform( size=(self.num_particles, self.particle_length))
   
        self.velocity_array = self.alpha * self.velocity_array + \
                       self.beta * cognitive_component_array * (self.particle_pbest - self.particle_array) + \
                       self.gamma * local_social_component_array * ( self.particle_ibest - self.particle_array) + \
                       self.delta * global_social_component_array * (self.gbest - self.particle_array)
        
        # --- MODIFICATION 1: Velocity Clamping Method ---
        self.velocity_array = np.clip(self.velocity_array, self.v_min, self.v_max)
        self.particle_array = self.particle_array + self.epsilon * self.velocity_array
        # Apply penalty to particle positions that are out of bounds
        penalty_mask = (self.particle_array > self.p_max) | (self.particle_array < self.p_min)
        self.particle_array = np.clip(self.particle_array, self.p_min, self.p_max)
        self.velocity_array[penalty_mask] *= -0.5
        
        
        # Check whether the current PSO implementation has discrete variables
        if self.discrete_params is not None:
            self._normalize_discrete_variables()
        
        # Calculate fitness values and update personal best array
        self.fitness_values_array = self.objective_function(self.particle_array)
        improved_particles_idx = self.fitness_values_array < self.personal_best_fitness_values
        self.particle_pbest[improved_particles_idx] = self.particle_array[improved_particles_idx]
        self.personal_best_fitness_values[improved_particles_idx] = self.fitness_values_array[improved_particles_idx]
        
        # Update global best position and value
        global_best_idx = np.argmin(self.personal_best_fitness_values)
        if self.personal_best_fitness_values[global_best_idx] < self.gbest_value:
            self.gbest = self.particle_array[global_best_idx]
            self.gbest_value = self.personal_best_fitness_values[global_best_idx]       
        
        self._update_informants_best()
        
        # --- Record for visualization --- genereated by chatgpt
        self.mean_fitness = np.mean(self.fitness_values_array)
        self.std_fitness = np.std(self.fitness_values_array)  
        self.mean_fitness_history.append(self.mean_fitness)
        self.std_fitness_history.append(self.std_fitness) 
        self.gbest_value_history.append(self.gbest_value)
        self.particle_history.append(self.particle_array.copy())
        self.velocity_history.append(self.velocity_array.copy())
        self.fitness_history.append(self.fitness_values_array.copy())
        self.gbest_position_history.append(self.gbest.copy())
        
        
    