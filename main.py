import numpy as np
import Utility.data_loader as data_loader
import ANN.ann as ann
import PSO.pso as pso
import Utility.ann_pso_bridge as ann_pso_bridge
import ANN.loss_functions as loss_functions 
import Utility.visualizer as v

# --- TUNED PARAMETERS ---
# 1. Back to the original, best-performing architecture
LAYERS = [8, 16, 16, 1] 
# 2. Massively increase particles for a "brute force" search
NUM_PARTICLES = 50
# 3. Increase iterations for a longer search
NUM_ITERATIONS = 100
NUM_INFORMANTS = 10
# 4. Back to the best-performing loss function
LOSS_FUNCTION = 'mse' 
# --- END TUNED PARAMETERS ---

PSO_PARAMS = {
    'alpha': 0.729,   
    'beta': 1.494,    
    'gamma': 1.494,     
    'delta': 0.1,   
    'epsilon': 0.75  
}

PSO_PARAMS_2 = {
    'alpha': 0.75,   
    'beta': 1.0,    
    'gamma': 1.0,     
    'delta': 0.0,   
    'epsilon': 0.75 
}

PSO_PARAMS = PSO_PARAMS_2

def main():
    # 2. Load Data
    (X_train, y_train), (X_test, y_test), y_scale_params = \
        data_loader.load_concrete_data("concrete_data.csv")
    
    y_mean, y_std = y_scale_params

    # 3. Create Model "Blueprint"
    model_template = ann.MultiLayerANN(layers=LAYERS)

    # 4. Use Bridge to Create PSO-Specific Components
    initial_particles, particle_length, discrete_params = ann_pso_bridge.initialize_particles(model_template, NUM_PARTICLES)
    
    obj_func = ann_pso_bridge.create_objective_function(
        model_template, 
        X_train, 
        y_train, 
        loss_function_name=LOSS_FUNCTION
    )

    # 5. Initialize and Run Optimizer
    optimizer = pso.ParticleSwarm(
        num_particles=NUM_PARTICLES,
        num_informants=NUM_INFORMANTS,
        num_iterations=NUM_ITERATIONS,
        objective_function=obj_func,
        particle_length=particle_length,
        discrete_params=discrete_params,
        particles=initial_particles,
        **PSO_PARAMS
    )
    

    # 6. Run Optimization
    print(f"Starting PSO optimization using {LOSS_FUNCTION.upper()}...") 
    final_scaled_loss = 0
    for i in range(NUM_ITERATIONS):
        optimizer._update()
        
        scaled_loss = optimizer.Gbest_value
        scaled_mean_loss = optimizer.mean_fitness
        real_loss = 0
        real_mean_loss = 0
        
        # This logic is CRITICAL for correct reporting
        if LOSS_FUNCTION == 'mse':
            real_loss = scaled_loss * (y_std ** 2)
            real_mean_loss = scaled_mean_loss * (y_std ** 2)
        elif LOSS_FUNCTION == 'rmse' or LOSS_FUNCTION == 'mae':
            # Un-scale RMSE or MAE: (scaled_loss * std)
            real_loss = scaled_loss * y_std
            real_mean_loss = scaled_mean_loss * y_std
        else:
            real_loss = scaled_loss # Default for unknown
            real_mean_loss = scaled_mean_loss # Default for unknown
        
        final_real_loss = real_loss
            
        # 7. Print progress periodically
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{NUM_ITERATIONS}, Global Best Loss: {real_loss:.6f}, Mean Loss: {real_mean_loss}")

    print("\nOptimization Finished.")
    
    # 8. Un-scale final report
    intial_scaled_loss = optimizer.Gbest_value_history[0] # Get the last recorded loss
    final_scaled_loss = optimizer.Gbest_value_history[-1] # Get the last recorded loss
    print(f"Initial Best Loss (scaled): {intial_scaled_loss:.6f}")
    print(f"Final Best Loss (scaled): {final_scaled_loss:.6f}")
    print(f"Final Best Loss (un-scaled): {final_real_loss:.6f}")


    # 9. Evaluate on Test Set
    best_params = ann_pso_bridge.reconstruct_params(optimizer.Gbest, model_template)
    test_predictions_scaled = model_template.evaluate_with_params(X_test, best_params)
    
    # Un-scale predictions and test values
    test_predictions_real = (test_predictions_scaled * y_std) + y_mean
    y_test_real = (y_test * y_std) + y_mean
    
    # Calculate the correct test loss
    test_loss_func = loss_functions.get_loss_function(LOSS_FUNCTION)
    test_real_loss = test_loss_func(y_test_real, test_predictions_real)
    
    print(f"Test Set {LOSS_FUNCTION.upper()} (scaled): {test_real_loss:.6f}")
    
    # Visualizations
    visualizer = v.Visualizer(
        pso=optimizer,
        layers=LAYERS,
        pso_params=PSO_PARAMS,
        num_particles=NUM_PARTICLES,
        num_iterations=NUM_ITERATIONS,
        num_informants=NUM_INFORMANTS,
        loss_function=LOSS_FUNCTION,
        intial_scaled_loss = intial_scaled_loss,
        final_scaled_loss = final_scaled_loss,
        final_real_loss = final_real_loss,
        test_real_loss = test_real_loss
    )   
    
    visualizer.record_test()
    #visualizer.animate_pso_pca()


if __name__ == "__main__":
    main()

