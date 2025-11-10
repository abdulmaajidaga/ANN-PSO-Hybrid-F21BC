import numpy as np
import Utility.data_handler as data_handler_class
import ANN.ann as ann
import PSO.pso as pso
import BRIDGE.bridge as bridge
import ANN.loss_functions as loss_functions 
import Utility.visualizer as v
import Utility.model_utils as model_utils

# --- TUNED PARAMETERS ---
# 1. Back to the original, best-performing architecture
LAYERS = [8, 16, 16, 1] 
# 2. Massively increase particles for a "brute force" search
NUM_PARTICLES = 30
# 3. Increase iterations for a longer search
NUM_ITERATIONS = 1000
NUM_INFORMANTS = 6
# 4. Back to the best-performing loss function
LOSS_FUNCTION = 'mae' 
DISCRETE_PSO = True
# --- END TUNED PARAMETERS ---

PSO_PARAMS = {
    'alpha': 0.8,   
    'beta': 1.2,    
    'gamma': 1.5,     
    'delta': 0.5,   
    'epsilon': 0.8  
}

PSO_PARAMS_GLOBAL = {
    'alpha': 0.729,    
    'beta': 1.49445,   
    'gamma': 0.0,      
    'delta': 1.49445,  
    'epsilon': 1.0      
}

PSO_PARAMS_LOCAL = {
    'alpha': 0.729,   
    'beta': 1.49445,    
    'gamma': 1.49445, 
    'delta': 0.0,       
    'epsilon': 0.85    
}

PSO_PARAMS_HYBRID = {
    'alpha': 0.729,
    'beta': 1.49445,
    'gamma': 1.49445,
    'delta': 1.49445,
    'epsilon': 1.0
}

PSO_PARAMS = PSO_PARAMS_LOCAL

def main():
    # 2. Load Data
    data_handler = data_handler_class.DataHandler()
    (X_train_scaled, y_train), (X_test_scaled, y_test) = data_handler.transform_data(path="concrete_data.csv", train_split = 0.7, random_seed = 1)

    # 3. Create Model "Blueprint"
    model_template = ann.MultiLayerANN(layers=LAYERS)

    # 4. Use Bridge to Create PSO-Specific Components
    ann_pso_bridge = bridge.Bridge(model_template, discrete = DISCRETE_PSO )
    initial_particles, particle_length, discrete_params = ann_pso_bridge.initialize_particles(NUM_PARTICLES)
    obj_func = ann_pso_bridge.create_objective_function(
        X_train_scaled, 
        y_train, 
        loss_function_name=LOSS_FUNCTION,
    )

    # 5. Initialize and Run Optimizer
    optimizer = pso.ParticleSwarm(
        num_particles=NUM_PARTICLES,
        num_informants=NUM_INFORMANTS,
        objective_function=obj_func,
        particle_length=particle_length,
        discrete_params=discrete_params,
        particles=initial_particles,
        **PSO_PARAMS
    )
    

    # 6. Run Optimization
    print(f"Starting PSO optimization using {LOSS_FUNCTION.upper()}...\n") 
    for i in range(NUM_ITERATIONS):
        optimizer._update()
        real_loss = optimizer.gbest_value
        real_mean_loss = optimizer.mean_fitness
           
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{NUM_ITERATIONS}, Global Best Loss: {real_loss:.6f}, Mean Loss: {real_mean_loss}")

    print("\nOptimization Finished.")
    
    # 8. Un-scale final report
    intial_train_loss = optimizer.gbest_value_history[0] # Get the last recorded loss
    final_train_loss = optimizer.gbest_value_history[-1] # Get the last recorded loss
    print(f"Initial Best Loss: {intial_train_loss:.6f}")
    print(f"Final Best Loss: {final_train_loss:.6f}")

    # 9. Evaluate on Test Set
    best_params = ann_pso_bridge.reconstruct_params(optimizer.Gbest)
    y_test_predictions = model_template.evaluate_with_params(X_test_scaled, best_params)
    
    # Calculate the correct test loss
    test_loss_func = loss_functions.get_loss_function(LOSS_FUNCTION)
    test_loss = test_loss_func(y_test, y_test_predictions)
    
    print(f"Test Set {LOSS_FUNCTION.upper()}: {test_loss:.6f}\n")
    
    # Visualizations
    visualizer = v.Visualizer(
        pso=optimizer,
        layers=LAYERS,
        pso_params=PSO_PARAMS,
        num_particles=NUM_PARTICLES,
        num_iterations=NUM_ITERATIONS,
        num_informants=NUM_INFORMANTS,
        loss_function=LOSS_FUNCTION,
        train_loss = final_train_loss,
        test_loss = test_loss
    )   
    
    test_folder = visualizer.record_test()
    model_utils.plot_predictions(y_test, y_test_predictions, test_folder = test_folder)
    # model_utils.save_and_evaluate(optimizer, model_template, ann_pso_bridge, y_mean, y_std, PSO_PARAMS, NUM_ITERATIONS, LOSS_FUNCTION, X_test_scaled, y_test)



if __name__ == "__main__":
    main()

