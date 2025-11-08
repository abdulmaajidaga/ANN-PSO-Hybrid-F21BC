"""
Visualization module for NEAT-Python
This is a standalone version that works without graphviz installation
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.figure(figsize=(10, 6))
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    print(f"Fitness statistics plot saved to {filename}")
    if view:
        plt.show()
    plt.close()


def plot_species(statistics, view=False, filename='species_stats.svg'):
    """ Visualizes speciation throughout evolution. """
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)
    print(f"Species statistics plot saved to {filename}")
    if view:
        plt.show()
    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, 
             show_disabled=True, prune_unused=False, node_colors=None, fmt='svg'):
    """ 
    Draws a neural network using matplotlib instead of graphviz.
    This is a simplified version that doesn't require graphviz installation.
    """
    if filename is None:
        filename = 'network'
    
    # Remove extension if provided
    if filename.endswith('.svg') or filename.endswith('.png'):
        filename = filename.rsplit('.', 1)[0]
    
    print("\nDrawing network topology using matplotlib...")
    
    # Get nodes
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes.keys() if n not in input_nodes and n not in output_nodes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate positions
    layers = {'input': [], 'hidden': [], 'output': []}
    
    # Input layer
    for i, node_id in enumerate(sorted(input_nodes)):
        x = 0.1
        y = 1 - (i + 1) / (len(input_nodes) + 1)
        layers['input'].append((node_id, x, y))
    
    # Hidden layer
    for i, node_id in enumerate(sorted(hidden_nodes)):
        x = 0.5
        y = 1 - (i + 1) / (len(hidden_nodes) + 1) if hidden_nodes else 0.5
        layers['hidden'].append((node_id, x, y))
    
    # Output layer
    for i, node_id in enumerate(sorted(output_nodes)):
        x = 0.9
        y = 1 - (i + 1) / (len(output_nodes) + 1)
        layers['output'].append((node_id, x, y))
    
    # Create position dictionary
    pos = {}
    for layer_name, nodes in layers.items():
        for node_id, x, y in nodes:
            pos[node_id] = (x, y)
    
    # Draw connections
    for conn_key, conn in genome.connections.items():
        if conn.enabled or show_disabled:
            input_id, output_id = conn_key
            if input_id in pos and output_id in pos:
                x1, y1 = pos[input_id]
                x2, y2 = pos[output_id]
                
                color = 'green' if conn.weight > 0 else 'red'
                alpha = 1.0 if conn.enabled else 0.3
                linewidth = min(abs(conn.weight) * 2, 5)
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=linewidth, zorder=1)
    
    # Draw nodes
    for layer_name, nodes in layers.items():
        for node_id, x, y in nodes:
            if layer_name == 'input':
                color = 'lightgray'
                shape = 's'  # square
            elif layer_name == 'output':
                color = 'lightblue'
                shape = 'o'  # circle
            else:
                color = 'white'
                shape = 'o'  # circle
            
            ax.scatter(x, y, s=800, c=color, marker=shape, 
                      edgecolors='black', linewidths=2, zorder=2)
            
            # Add labels
            if node_names and node_id in node_names:
                label = node_names[node_id]
            else:
                label = str(node_id)
            
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold', zorder=3)
    
    plt.title('Neural Network Topology', fontsize=14, fontweight='bold')
    
    # Save figure
    output_file = f"{filename}.{fmt}"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Topology graph saved to {output_file}")
    
    if view:
        plt.show()
    plt.close()
    
    return None  # graphviz would return a Digraph object