import neat
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Import visualization tools from neat-python
try:
    import visualize
except ImportError:
    print("Warning: 'visualize' module not found.")
    print("Please install it for graph plotting: pip install graphviz")
    visualize = None

# Import the data loading function from the 'Utility' package
from Utility.data_handler import load_concrete_data

# --- Global Data ---
# Load data once to be used in the fitness function
print("Loading data...")
# The path is relative to the root folder where you run the command
(X_train, y_train), (X_test, y_test), y_scale_params = load_concrete_data(path="concrete_data.csv")
y_mean, y_std = y_scale_params

# Ravel y_train for fitness calculation (from (n, 1) to (n,))
y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

print(f"Data loaded: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
print(f"Input features: {X_train.shape[1]}")


def eval_genomes(genomes, config):
    """
    Fitness function.
    Evaluates each genome's performance on the regression task.
    """
    for genome_id, genome in genomes:
        # Create a feed-forward network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        predictions = []
        for xi in X_train:
            output = net.activate(xi)
            predictions.append(output[0]) # We have 1 output node
        
        predictions = np.array(predictions)
        
        try:
            mse = np.mean((predictions - y_train_flat) ** 2)
            if not np.isfinite(mse):
                genome.fitness = -1.0
            else:
                genome.fitness = 1.0 / (1.0 + mse)
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            genome.fitness = -1.0 

def run_evolution(config_path, generations):
    """
    Sets up and runs the NEAT evolution process.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    # Add reporters to show progress in the console
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Checkpoints will be saved in a 'neat-checkpoints' folder in your root
    p.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoints/chk-'))

    print("\n--- Running Evolution ---")
    winner = p.run(eval_genomes, generations)

    print(f"\nEvolution complete. Best genome found:")
    print(winner)

    # Save the Winner to the root folder
    winner_path = 'winner.pkl'
    with open(winner_path, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Best genome saved to {winner_path}")

    if visualize:
        visualize.plot_stats(stats, ylog=False, view=False, filename="fitness_stats.svg")
        visualize.plot_species(stats, view=False, filename="species_stats.svg")
    else:
        print("\nSkipping statistics plotting (visualize module not found).")
    
    print("\n--- Testing Winner Network ---")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Evaluate on Training Data
    train_predictions_scaled = []
    for xi in X_train:
        output = winner_net.activate(xi)
        train_predictions_scaled.append(output[0])
    train_predictions_scaled = np.array(train_predictions_scaled)
    train_mse_scaled = np.mean((train_predictions_scaled - y_train_flat) ** 2)
    print(f"Winner Training MSE (scaled): {train_mse_scaled:.6f}")
    
    # Evaluate on Test Data (Unseen)
    test_predictions_scaled = []
    for xi in X_test:
        output = winner_net.activate(xi)
        test_predictions_scaled.append(output[0])
    test_predictions_scaled = np.array(test_predictions_scaled)
    test_mse_scaled = np.mean((test_predictions_scaled - y_test_flat) ** 2)
    print(f"Winner Test MSE (scaled):     {test_mse_scaled:.6f}")

    print("\nSample Test Predictions (Un-scaled):")
    print("  Actual (raw) | Predicted (raw)")
    print("-" * 30)
    
    y_test_raw = (y_test_flat * y_std) + y_mean
    test_predictions_raw = (test_predictions_scaled * y_std) + y_mean
    
    for i in range(min(10, len(y_test_raw))):
        print(f" {y_test_raw[i]:>12.2f} | {test_predictions_raw[i]:>15.2f}")
    
    test_rmse_raw = np.sqrt(np.mean((test_predictions_raw - y_test_raw) ** 2))
    print("-" * 30)
    print(f"Winner Test RMSE (un-scaled): {test_rmse_raw:.4f}")

    return winner

if __name__ == '__main__':
    # Determine path to configuration file (it's in the same folder as this script)
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-neat.txt')
    
    # Create checkpoint directory if it doesn't exist in the root
    if not os.path.exists('neat-checkpoints'):
        os.makedirs('neat-checkpoints')
        
    run_evolution(config_file, generations=50)
