import neat
import pickle
import os
import argparse

try:
    import visualize
except ImportError:
    print("Error: 'visualize' module not found.")
    print("Please install it for graph plotting: pip install graphviz")
    print("You may also need to install the graphviz system package.")
    visualize = None

def draw_best_topology(config, genome, view=True, filename='best_topology.svg'):
    """
    Generates a visualization of the winning network topology.
    """
    if not visualize:
        print("\nCannot draw topology (visualize module not loaded).")
        return
        
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    
    node_names = {-i-1: f'IN {i}' for i in range(num_inputs)}
    for i in range(num_outputs):
        node_names[i] = f'OUT {i}'

    print("Drawing network topology...")
    try:
        visualize.draw_net(config, genome, view=view, 
                           node_names=node_names,
                           filename=filename,
                           show_disabled=True)
        print(f"Topology graph saved to {filename}")
    except Exception as e:
        print("\n--- Error during visualization ---")
        print(f"Failed to draw network: {e}")
        print("This often means the 'graphviz' system library is not installed or not in your PATH.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a saved NEAT genome.')
    
    parser.add_argument('--genome', type=str, 
                        help='Path to the pickled winner genome file.', 
                        default='winner.pkl')
                        
    parser.add_argument('--config', type=str, 
                        help='Path to the NEAT config file.', 
                        default=None) # Default to None, we will find it
                        
    args = parser.parse_args()

    # --- Determine Config Path ---
    # This is the updated logic to find the config file
    local_dir = os.path.dirname(__file__)
    if args.config is None:
        # If no config is given, assume it's next to this script
        config_path = os.path.join(local_dir, 'config-neat.txt')
    else:
        config_path = args.config

    # --- Load Files ---
    try:
        # 'winner.pkl' is saved in the root, so this path is correct
        with open(args.genome, 'rb') as f:
            winner_genome = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Genome file not found at '{args.genome}'")
        print("Please run 'python -m NEAT.evolve' first to generate a winner.")
        exit()
        
    try:
        # Load config from the path we just determined
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        exit()

    # --- Draw the network ---
    draw_best_topology(config, winner_genome)
