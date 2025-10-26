# pso.py
import numpy as np

class ParticleSwarm(object):
    def  __init__(self, num_particles, num_informants,  particle_length, objective_function, num_iterations,
                 alpha = 0.9, beta = 0.8, gamma = 0.5, delta = 0.8, epsilon = 0.25, particles = None, v_max_scale = 0.2):
        
        self.num_particles = num_particles 
        self.particle_length = particle_length 
        self.objective_function = objective_function 
        
        self.alpha_start = alpha
        self.alpha_end = 0.4  # Common final value for inertia
        self.alpha_decay_step = (self.alpha_start - self.alpha_end) / num_iterations
        self.current_alpha = self.alpha_start
        
        self.beta = beta 
        self.gamma = gamma 
        self.delta = delta 
        self.epsilon = epsilon 
        self.num_informants = num_informants
        
        if particles is not None:
            self.particle_array = particles
        else:
            self.particle_array = np.random.uniform(size=(num_particles, particle_length)) 

        self.velocity_array = np.random.rand(num_particles, particle_length) 
        
        p_min = self.particle_array.min()
        p_max = self.particle_array.max()
        self.v_max = (p_max - p_min) * v_max_scale
        self.v_min = -self.v_max

        self.personal_best_array = np.copy(self.particle_array) 
        
        self.fitness_values_array =  self.objective_function(self.particle_array)
        self.personal_best_fitness_values = np.copy(self.fitness_values_array) 
        
        self.informants_best_array = np.zeros_like(self.particle_array)
        
        global_best_idx = np.argmin(self.fitness_values_array)
        self.Gbest = self.particle_array[global_best_idx] 
        self.Gbest_value = self.fitness_values_array[global_best_idx] 
        print(f"Initial Global Best Value: {self.Gbest_value}")
        self.particle_history = [np.copy(self.particle_array)]
        

    
    # --- MODIFICATION 3: Renamed/Refactored Method ---
    def _update_informants_best(self):
        """
        Calculates the best informant for each particle *for this iteration*.
        This uses the *personal best* positions of the informants as the
        source of social knowledge.
        """
        for i in range(self.num_particles):
            # Select random unique indices excluding the current particle
            possible_indices = [j for j in range(self.num_particles) if j != i]
            informant_indices = np.random.choice(possible_indices,
                                                size=min(self.num_informants, len(possible_indices)),
                                                replace=False)
            
            # Get the *personal best fitness* of these informants
            informant_pbest_fitness = self.personal_best_fitness_values[informant_indices]

            # Find the best one *among the informants*
            best_local_idx_in_subset = np.argmin(informant_pbest_fitness)
            
            # Get the global index of that best informant
            best_global_idx = informant_indices[best_local_idx_in_subset]

            # Store the *position* of that best informant's pbest
            self.informants_best_array[i] = self.personal_best_array[best_global_idx]
    # --- END MODIFICATION 3 ---
            
            
    def _update(self):
        
        # --- MODIFICATION 4: Call new methods ---
        # 1. Update informant social network dynamically
        self._update_informants_best() 
        
        # 2. Update inertia weight
        self.current_alpha = max(self.alpha_end, self.current_alpha - self.alpha_decay_step)
        # --- END MODIFICATION 4 ---

        cognitive_component_array =  np.random.uniform(low = 0, high = self.beta, size=(self.num_particles, self.particle_length))
        local_social_component_array = np.random.uniform(low = 0, high = self.gamma, size=(self.num_particles, self.particle_length))
        global_social_component_array = np.random.uniform(low = 0, high = self.delta, size=(self.num_particles, self.particle_length))

        # Use self.current_alpha
        new_velocity = self.current_alpha * self.velocity_array + \
                       cognitive_component_array * (self.personal_best_array - self.particle_array) + \
                       local_social_component_array * ( self.informants_best_array - self.particle_array) + \
                       global_social_component_array * (self.Gbest - self.particle_array)
        
        self.velocity_array = np.clip(new_velocity, self.v_min, self.v_max)
        
        self.particle_array = self.particle_array + self.epsilon * self.velocity_array
        
        self.fitness_values_array = self.objective_function(self.particle_array)
        
        improved_particles_idx = self.fitness_values_array < self.personal_best_fitness_values
        
        self.personal_best_array[improved_particles_idx] = self.particle_array[improved_particles_idx]
        self.personal_best_fitness_values[improved_particles_idx] = self.fitness_values_array[improved_particles_idx]
        
        global_best_idx = np.argmin(self.fitness_values_array)
        if self.fitness_values_array[global_best_idx] < self.Gbest_value:
            self.Gbest = self.particle_array[global_best_idx]
            self.Gbest_value = self.fitness_values_array[global_best_idx]
        self.particle_history.append(np.copy(self.particle_array))