# pso.py
import numpy as np


class ParticleSwarm(object):
    def  __init__(self, num_particles, num_informants, num_iterations,  objective_function,  particle_length, discrete_params = None,
                 alpha = 0.729, beta = 1.494, gamma = 1.0, delta = 1.494, epsilon = 0.15, particles = None, v_max_scale = 0.2):
        
        print(alpha)
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
        self.Gbest_value_history = []
        self.mean_fitness_history = []
        self.particle_history = []
        self.velocity_history = []
        self.fitness_history = []
        self.Gbest_position_history = []
        
        if particles is not None:
            self.particle_array = particles # Initialise with existing particles 
        else:
            self.particle_array = np.random.uniform(size=(num_particles, self.particle_length)) # Initialise with random particle array

        # Here, the particle velocities are randomly initialized in a new array, where each velocity array index matches the particle array index
        # For example, particle_array[0] has a velocity of velocity_array[0]
        self.velocity_array = np.random.rand(num_particles, self.particle_length) # Stores particle velocities at each particle index
        
        # --- MODIFICATION 1: Velocity Clamping Method ---
        # p_min = self.particle_array.min()
        # p_max = self.particle_array.max()
        # self.v_max = (p_max - p_min) * v_max_scale
        # self.v_min = -self.v_max
        # --- END MODIFICATION 1 ---

        # Stores personal best positions for each particle in the matching index
        self.personal_best_array = np.copy(self.particle_array)
        
        # Stores fitness values for each particle in the matching index
        self.fitness_values_array =  self.objective_function(self.particle_array) 
        self.personal_best_fitness_values = np.copy(self.fitness_values_array) 
        
        # Store informant best positions for each particle, where each informant best position matches the particle array index
        # For example,  particle_informant_best[0] stores the best informant vector position for particle_array[0]
        # and particle_informant_indices[0] contains the list of selected informants for particle_array[0] 
        self.particle_informant_indices, self.particle_informant_best = self.create_particle_informant_best_nearest(num_informants)
        
        
        # Find and store global best position and value
        global_best_idx = np.argmin(self.fitness_values_array)
        self.Gbest = self.particle_array[global_best_idx] 
        self.Gbest_value = self.fitness_values_array[global_best_idx] 
        print(f"Initial Global Best Value: {self.Gbest_value}")
        
        mean_fitness = np.mean(self.fitness_values_array)
        self.mean_fitness_history.append(mean_fitness)
        self.Gbest_value_history.append(self.Gbest_value)
        self.particle_history.append(self.particle_array.copy())
        self.velocity_history.append(self.velocity_array.copy())
        self.fitness_history.append(self.fitness_values_array.copy())
        self.Gbest_position_history.append(self.Gbest.copy())

        

    def create_particle_informant_best_nearest(self, num_informants):
        num_particles = self.num_particles
        particle_array = self.particle_array
        particle_informant_best = np.zeros_like(particle_array)
        particle_informant_indices = np.zeros((num_particles, num_informants), dtype=int)

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
            particle_informant_indices[i] = nearest_indices
            # Get the *personal best fitness* of these informants
            informant_pbest_values = self.personal_best_fitness_values[nearest_indices]
            # Find the best one *among the informants*
            best_local_idx = np.argmin(informant_pbest_values)
            # Get the global index of that best informant
            best_global_idx = nearest_indices[best_local_idx]
            # Store the *position* of that best informant's pbest
            particle_informant_best[i] = self.personal_best_array[best_global_idx]
            

        return particle_informant_indices, particle_informant_best
      
    def _update_informants_best(self):
        """
        Calculates the best informant for each particle *for this iteration*.
        This uses the *personal best* positions of the informants as the
        source of social knowledge.
        """
        for i in range(self.num_particles):
            # Get the *personal best fitness* of these informants
            informants_list = self.particle_informant_indices[i]
            informant_pbest_values = self.personal_best_fitness_values[informants_list]
            # Find the best one *among the informants*
            best_local_idx = np.argmin(informant_pbest_values)
            # Get the global index of that best informant
            best_global_idx = informants_list[best_local_idx]
            # Store the *position* of that best informant's pbest
            self.particle_informant_best[i] = self.personal_best_array[best_global_idx]
            
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
    
        cognitive_component_array =  np.random.uniform(low = 0, high = self.beta, size=(self.num_particles, self.particle_length))
        local_social_component_array = np.random.uniform(low = 0, high = self.gamma, size=(self.num_particles, self.particle_length))
        global_social_component_array = np.random.uniform(low = 0, high = self.delta, size=(self.num_particles, self.particle_length))
   
        self.velocity_array = self.alpha * self.velocity_array + \
                       cognitive_component_array * (self.personal_best_array - self.particle_array) + \
                       local_social_component_array * ( self.particle_informant_best - self.particle_array) + \
                       global_social_component_array * (self.Gbest - self.particle_array)
        
        # --- MODIFICATION 1: Velocity Clamping Method ---
        #self.velocity_array = np.clip(new_velocity, self.v_min, self.v_max)
        
        self.particle_array = self.particle_array + self.epsilon * self.velocity_array
        
        # Check whether the current PSO implementation has discrete variables
        if self.discrete_params is not None:
            self._normalize_discrete_variables()
        
        # Calculate fitness values and update personal best array
        self.fitness_values_array = self.objective_function(self.particle_array)
        improved_particles_idx = self.fitness_values_array < self.personal_best_fitness_values
        self.personal_best_array[improved_particles_idx] = self.particle_array[improved_particles_idx]
        self.personal_best_fitness_values[improved_particles_idx] = self.fitness_values_array[improved_particles_idx]
        
        ## Version to set gbest based on only the particles personal best fitness values
        # Update global best position and value
        global_best_idx = np.argmin(self.personal_best_fitness_values)
        if self.personal_best_fitness_values[global_best_idx] < self.Gbest_value:
            self.Gbest = self.particle_array[global_best_idx]
            self.Gbest_value = self.personal_best_fitness_values[global_best_idx]

            
        # ## Version to set gbest based on the current particle fitness values
        # # Update global best position and value
        # global_best_idx = np.argmin(self.fitness_values_array)
        # if self.fitness_values_array[global_best_idx] < self.Gbest_value:
        #     self.Gbest = self.particle_array[global_best_idx]
        #     self.Gbest_value = self.fitness_values_array[global_best_idx]
            
        
        self._update_informants_best()
        
        # --- Record for visualization --- genereated by chatgpt
        self.mean_fitness = np.mean(self.fitness_values_array)
        self.mean_fitness_history.append(self.mean_fitness)
        self.Gbest_value_history.append(self.Gbest_value)
        self.particle_history.append(self.particle_array.copy())
        self.velocity_history.append(self.velocity_array.copy())
        self.fitness_history.append(self.fitness_values_array.copy())
        self.Gbest_position_history.append(self.Gbest.copy())
        #self.particle_history.append(np.copy(self.particle_array))
        
        
    