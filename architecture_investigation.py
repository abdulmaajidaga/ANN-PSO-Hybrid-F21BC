"""
Complete ANN Architecture Investigation
Single file that runs experiments and generates analysis
(Refactored for clarity and modularity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Your existing imports
import Utility.data_handler as data_handler
import ANN.ann as ann
import PSO.pso as pso
from BRIDGE import initialize_particles, create_objective_function, reconstruct_params
import ANN.loss_functions as loss_functions


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base PSO parameters (kept constant across all experiments)
BASE_PSO_PARAMS = {
    'alpha': 0.729,
    'beta': 1.494,
    'gamma': 1.494,
    'delta': 0.1,
    'epsilon': 0.75
}

# PSO configuration
PSO_CONFIG = {
    'num_particles': 50,
    'num_iterations': 500,
    'num_informants': 10,
    'loss_function': 'rmse'
}

# Experiment settings
NUM_RUNS = 10  # How many times to repeat each configuration
DATASET_PATH = "concrete_data.csv"

# --- NEW: Control which experiments to run ---
RUN_EXPERIMENT = {
    'depth': True,
    'width': True,
    'shape': True,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _count_parameters(layers):
    """Calculates the total number of weights and biases in the ANN."""
    return sum(layers[i] * layers[i+1] + layers[i+1] 
               for i in range(len(layers) - 1))

def _unscale_loss(scaled_loss, y_std, loss_function_name):
    """Unscales a loss value based on the loss function used."""
    if loss_function_name == 'mse':
        return scaled_loss * (y_std ** 2)
    if loss_function_name in ['rmse', 'mae']:
        return scaled_loss * y_std
    return scaled_loss

def _find_convergence_iteration(history, threshold_percent=0.95):
    """Finds iteration where 95% of total improvement is reached."""
    if not history or len(history) < 2:
        return 0
    
    initial_loss = history[0]
    final_loss = history[-1]
    
    # Handle no improvement
    if initial_loss <= final_loss:
        return 0
        
    total_improvement = initial_loss - final_loss
    threshold = initial_loss - (total_improvement * threshold_percent)
    
    for i, loss in enumerate(history):
        if loss <= threshold:
            return i
            
    return len(history) - 1  # Reached the end


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(architecture, X_train, y_train, X_test, y_test, y_scale_params, run_id):
    """Run a single experiment with given architecture"""
    
    print(f"\n{'='*60}")
    print(f"Run {run_id}: {architecture['layers']}")
    print(f"{'='*60}")
    
    y_mean, y_std = y_scale_params
    
    # Create model
    model = ann.MultiLayerANN(layers=architecture['layers'])
    
    # Initialize PSO components
    initial_particles, particle_length, discrete_params = \
        initialize_particles(model, PSO_CONFIG['num_particles'])
    
    obj_func = create_objective_function(
        model, 
        X_train, 
        y_train, 
        loss_function_name=PSO_CONFIG['loss_function']
    )
    
    # Create optimizer
    optimizer = pso.ParticleSwarm(
        num_particles=PSO_CONFIG['num_particles'],
        num_informants=PSO_CONFIG['num_informants'],
        num_iterations=PSO_CONFIG['num_iterations'],
        objective_function=obj_func,
        particle_length=particle_length,
        discrete_params=discrete_params,
        particles=initial_particles,
        **BASE_PSO_PARAMS
    )
    
    # Run optimization
    for i in range(PSO_CONFIG['num_iterations']):
        optimizer._update()
        
        if (i + 1) % 100 == 0:
            scaled_loss = optimizer.Gbest_value
            # Use helper function
            real_loss = _unscale_loss(scaled_loss, y_std, PSO_CONFIG['loss_function'])
            print(f"   Iter {i+1}: Loss = {real_loss:.6f}")
    
    # Get results
    initial_loss_scaled = optimizer.Gbest_value_history[0]
    final_loss_scaled = optimizer.Gbest_value_history[-1]
    
    # Unscale losses using helper function
    initial_loss = _unscale_loss(initial_loss_scaled, y_std, PSO_CONFIG['loss_function'])
    final_loss = _unscale_loss(final_loss_scaled, y_std, PSO_CONFIG['loss_function'])
    
    # Test set evaluation
    best_params = reconstruct_params(optimizer.Gbest, model)
    test_predictions_scaled = model.evaluate_with_params(X_test, best_params)
    test_predictions_real = (test_predictions_scaled * y_std) + y_mean
    y_test_real = (y_test * y_std) + y_mean
    
    test_loss_func = loss_functions.get_loss_function(PSO_CONFIG['loss_function'])
    test_loss = test_loss_func(y_test_real, test_predictions_real)
    
    # Count parameters using helper function
    total_params = _count_parameters(architecture['layers'])
    
    # Find convergence iteration using helper function
    convergence_iteration = _find_convergence_iteration(optimizer.Gbest_value_history)
    
    # Calculate improvement percentage
    improvement_percent = 0.0
    if initial_loss > 0: # Avoid division by zero
        improvement_percent = ((initial_loss - final_loss) / initial_loss) * 100

    return {
        'run_id': run_id,
        'layers': str(architecture['layers']),
        'num_layers': len(architecture['layers']),
        'total_params': total_params,
        'initial_train_loss': initial_loss,
        'final_train_loss': final_loss,
        'test_loss': test_loss,
        'improvement_percent': improvement_percent,
        'convergence_iteration': convergence_iteration,
        'generalization_gap': test_loss - final_loss,
    }

# ============================================================================
# EXPERIMENT DEFINITIONS
# (These functions now *define* experiments, they don't run them)
# ============================================================================

def _define_depth_architectures():
    """Returns configuration for Experiment 1: Effect of network depth"""
    print("...Defining Depth Experiment Architectures")
    base_width = 16
    depths = [2, 3, 4, 5]
    architectures = []
    
    for depth in depths:
        if depth == 2:
            layers = [8, 1]  # No hidden layers
        else:
            layers = [8] + [base_width] * (depth - 2) + [1]
        
        architectures.append({
            'layers': layers,
            'depth': depth
        })
    return architectures

def _define_width_architectures():
    """Returns configuration for Experiment 2: Effect of network width"""
    print("...Defining Width Experiment Architectures")
    num_hidden_layers = 2
    widths = [4, 8, 16, 32, 64]
    architectures = []
    
    for width in widths:
        layers = [8] + [width] * num_hidden_layers + [1]
        architectures.append({
            'layers': layers,
            'width': width
        })
    return architectures

def _define_shape_architectures():
    """Returns configuration for Experiment 3: Effect of architecture shape"""
    print("...Defining Shape Experiment Architectures")
    shape_configs = [
        {'name': 'pyramid_expanding', 'layers': [8, 16, 32, 64, 1]},
        {'name': 'pyramid_contracting', 'layers': [8, 64, 32, 16, 1]},
        {'name': 'diamond', 'layers': [8, 16, 32, 16, 1]},
        {'name': 'rectangle', 'layers': [8, 16, 16, 16, 1]},
        {'name': 'bottleneck', 'layers': [8, 32, 8, 32, 1]},
    ]
    # Convert to the format expected by the runner
    return [{'layers': arch['layers'], 'shape': arch['name']} for arch in shape_configs]


def _run_experiment_set(experiment_name, architectures, data_inputs):
    """
    NEW: This generic helper function runs a set of experiments,
    handling all the repetitive looping and result metadata tagging.
    """
    print(f"\n{'='*70}\nRUNNING EXPERIMENT: {experiment_name.upper()}\n{'='*70}")
    
    (X_train, y_train), (X_test, y_test), y_scale_params = data_inputs
    results_list = []
    
    for arch_info in architectures:
        architecture = {'layers': arch_info['layers']}
        
        for run in range(NUM_RUNS):
            result = run_single_experiment(
                architecture, 
                X_train, y_train, X_test, y_test, 
                y_scale_params, 
                run
            )
            
            # Add all extra metadata (e.g., 'depth', 'width', 'shape')
            # We copy and remove 'layers' to avoid the unhashable list error
            metadata = arch_info.copy()
            metadata.pop('layers', None)
            
            result.update(metadata)
            result['experiment'] = experiment_name
            results_list.append(result)
            
    return results_list


# ============================================================================
# ANALYSIS AND VISUALIZATION (Refactored)
# ============================================================================

def _plot_depth_analysis(depth_df):
    """Analyzes and plots results for the depth experiment."""
    print("\n--- Depth Analysis ---")
    summary = depth_df.groupby('depth').agg({
        'final_train_loss': ['mean', 'std'],
        'test_loss': ['mean', 'std'],
        'convergence_iteration': 'mean',
        'total_params': 'first'
    }).round(4)
    print(summary)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect of Network Depth', fontsize=16, fontweight='bold')
    
    # Plot 1: Training vs Test Loss (Using Seaborn)
    df_melt = depth_df.groupby('depth').agg(
        Training_Loss=('final_train_loss', 'mean'),
        Test_Loss=('test_loss', 'mean')
    ).reset_index().melt('depth', var_name='Loss Type', value_name='Loss')
    
    sns.barplot(data=df_melt, x='depth', y='Loss', hue='Loss Type', ax=axes[0, 0])
    axes[0, 0].set_title('Training vs Test Loss by Depth')
    axes[0, 0].set_xlabel('Number of Layers')
    axes[0, 0].set_ylabel('Loss (RMSE)')
    
    # Plot 2: Box plot (Using Seaborn)
    sns.barplot(data=depth_df, x='depth', y='test_loss', ax=axes[0, 1], estimator=np.mean, color='skyblue')
    axes[0, 1].set_title('Mean Test Loss by Depth')
    axes[0, 1].set_xlabel('Number of Layers')
    axes[0, 1].set_ylabel('Mean Test Loss (RMSE)')
    
    # Plot 3: Convergence speed
    sns.barplot(data=depth_df, x='depth', y='convergence_iteration', ax=axes[1, 0], color='coral', estimator=np.mean)
    axes[1, 0].set_title('Convergence Speed by Depth')
    axes[1, 0].set_xlabel('Number of Layers')
    axes[1, 0].set_ylabel('Iterations to 95% Convergence')
    
    # Plot 4: Parameters vs Performance
    param_perf = depth_df.groupby('depth').agg(
        total_params=('total_params', 'first'),
        test_loss=('test_loss', 'mean')
    ).reset_index()
    
    sns.scatterplot(data=param_perf, x='total_params', y='test_loss', s=100, ax=axes[1, 1])
    for _, row in param_perf.iterrows():
        axes[1, 1].annotate(f"{int(row['depth'])} layers", 
                          (row['total_params'], row['test_loss']),
                          xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Total Parameters')
    axes[1, 1].set_ylabel('Mean Test Loss')
    axes[1, 1].set_title('Model Complexity vs Performance')
    
    plt.tight_layout()
    plt.savefig('depth_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Depth analysis saved as 'depth_analysis.png'")
    plt.show()

def _plot_width_analysis(width_df):
    """Analyzes and plots results for the width experiment."""
    print("\n--- Width Analysis ---")
    summary = width_df.groupby('width').agg({
        'final_train_loss': ['mean', 'std'],
        'test_loss': ['mean', 'std'],
        'total_params': 'first'
    }).round(4)
    print(summary)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect of Network Width', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance by width
    df_melt = width_df.groupby('width').agg(
        Training_Loss=('final_train_loss', 'mean'),
        Test_Loss=('test_loss', 'mean')
    ).reset_index().melt('width', var_name='Loss Type', value_name='Loss')
    
    sns.lineplot(data=df_melt, x='width', y='Loss', hue='Loss Type', marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Performance by Network Width')
    axes[0, 0].set_xlabel('Neurons per Hidden Layer')
    axes[0, 0].set_ylabel('Loss (RMSE)')
    axes[0, 0].grid(True)
    
    # Plot 2: Generalization gap (Using Seaborn)
    width_df['gen_gap'] = width_df['test_loss'] - width_df['final_train_loss']
    sns.barplot(data=width_df, x='width', y='gen_gap', ax=axes[0, 1], color='salmon', estimator=np.mean)
    axes[0, 1].set_title('Generalization Gap by Width')
    axes[0, 1].set_xlabel('Neurons per Hidden Layer')
    axes[0, 1].set_ylabel('Test Loss - Train Loss')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Plot 3: Violin plot
    sns.barplot(data=width_df, x='width', y='test_loss', ax=axes[1, 0], estimator=np.mean, color='skyblue')
    axes[1, 0].set_title('Mean Test Loss by Width')
    axes[1, 0].set_xlabel('Neurons per Hidden Layer')
    axes[1, 0].set_ylabel('Mean Test Loss (RMSE)')
    
    # Plot 4: Parameters vs performance
    param_summary = width_df.groupby('width').agg(
        total_params=('total_params', 'first'),
        test_loss_mean=('test_loss', 'mean'),
        test_loss_std=('test_loss', 'std')
    ).reset_index()

    axes[1, 1].errorbar(param_summary['total_params'], 
                       param_summary['test_loss_mean'],
                       yerr=param_summary['test_loss_std'],
                       fmt='-o', capsize=5, linewidth=2) # Use fmt='-o' for line + marker
    axes[1, 1].set_xlabel('Total Parameters')
    axes[1, 1].set_ylabel('Mean Test Loss')
    axes[1, 1].set_title('Complexity vs Performance Trade-off')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('width_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Width analysis saved as 'width_analysis.png'")
    plt.show()

def _plot_shape_analysis(shape_df):
    """Analyzes and plots results for the shape experiment."""
    print("\n--- Shape Analysis ---")
    summary = shape_df.groupby('shape').agg({
        'test_loss': ['mean', 'std'],
        'convergence_iteration': 'mean',
        'layers': 'first'
    }).round(4)
    print(summary)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Effect of Architecture Shape', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance comparison (Using Seaborn)
    sns.barplot(data=shape_df, y='shape', x='test_loss', ax=axes[0], color='skyblue', estimator=np.mean)
    axes[0].set_title('Performance by Architecture Shape')
    axes[0].set_xlabel('Test Loss (RMSE)')
    axes[0].set_ylabel('Architecture Shape')
    
    # Plot 2: Convergence comparison (Using Seaborn)
    sns.barplot(data=shape_df, y='shape', x='convergence_iteration', ax=axes[1], color='lightcoral', estimator=np.mean)
    axes[1].set_title('Convergence Speed by Shape')
    axes[1].set_xlabel('Iterations to 95% Convergence')
    axes[1].set_ylabel('Architecture Shape')
    
    plt.tight_layout()
    plt.savefig('shape_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Shape analysis saved as 'shape_analysis.png'")
    plt.show()


def analyze_and_plot(results_df):
    """
    Main analysis dispatcher.
    Calls specific plotting functions based on experiments found in results.
    """
    print("\n" + "="*70)
    print("ANALYSIS AND VISUALIZATION")
    print("="*70)
    
    sns.set_style("whitegrid")
    
    # Suppress warnings from seaborn when data has no variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        if 'depth' in results_df['experiment'].values:
            _plot_depth_analysis(results_df[results_df['experiment'] == 'depth'])
        
        if 'width' in results_df['experiment'].values:
            _plot_width_analysis(results_df[results_df['experiment'] == 'width'])
        
        if 'shape' in results_df['experiment'].values:
            _plot_shape_analysis(results_df[results_df['experiment'] == 'shape'])


# ============================================================================
# REPORT GENERATOR
# ============================================================================

def generate_report(results_df):
    """Generate text report"""
    
    with open('architecture_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("ANN ARCHITECTURE INVESTIGATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total experiments run: {len(results_df)}\n")
        f.write(f"Unique architectures: {results_df['layers'].nunique()}\n")
        f.write(f"Best test loss: {results_df['test_loss'].min():.6f}\n")
        f.write(f"Worst test loss: {results_df['test_loss'].max():.6f}\n")
        f.write(f"Mean test loss: {results_df['test_loss'].mean():.6f}\n")
        f.write(f"Std test loss: {results_df['test_loss'].std():.6f}\n\n")
        
        # Top 5 architectures
        f.write("TOP 5 ARCHITECTURES\n")
        f.write("-"*70 + "\n")
        top5 = results_df.nsmallest(5, 'test_loss')[[
            'layers', 'test_loss', 'total_params', 'convergence_iteration'
        ]]
        f.write(top5.to_string(index=False) + "\n\n")
        
        # Findings per experiment
        for exp in results_df['experiment'].unique():
            f.write(f"\n{exp.upper()} EXPERIMENT\n")
            f.write("-"*70 + "\n")
            exp_df = results_df[results_df['experiment'] == exp]
            # Use .describe() for a quick numerical summary
            f.write(exp_df.describe().to_string() + "\n\n")
    
    print("\n✓ Report saved as 'architecture_report.txt'")


# ============================================================================
# MAIN EXECUTION (Refactored)
# ============================================================================

def main():
    print("="*70)
    print("ANN ARCHITECTURE INVESTIGATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"   Dataset: {DATASET_PATH}")
    print(f"   Particles: {PSO_CONFIG['num_particles']}")
    print(f"   Iterations: {PSO_CONFIG['num_iterations']}")
    print(f"   Runs per config: {NUM_RUNS}")
    print(f"   Loss function: {PSO_CONFIG['loss_function']}")
    
    # Load data
    print("\nLoading data...")
    # data_inputs is a tuple: ((X_train, y_train), (X_test, y_test), y_scale_params)
    data_inputs = data_handler.load_concrete_data(DATASET_PATH)
    
    # Run experiments
    all_results = []
    generated_files = ['architecture_results.csv', 'architecture_report.txt']
    
    if RUN_EXPERIMENT['depth']:
        depth_archs = _define_depth_architectures()
        all_results.extend(_run_experiment_set('depth', depth_archs, data_inputs))
        generated_files.append('depth_analysis.png')
    
    if RUN_EXPERIMENT['width']:
        width_archs = _define_width_architectures()
        all_results.extend(_run_experiment_set('width', width_archs, data_inputs))
        generated_files.append('width_analysis.png')
        
    if RUN_EXPERIMENT['shape']:
        shape_archs = _define_shape_architectures()
        all_results.extend(_run_experiment_set('shape', shape_archs, data_inputs))
        generated_files.append('shape_analysis.png')

    if not all_results:
        print("\nNo experiments were selected to run. Exiting.")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv('architecture_results.csv', index=False)
    print(f"\n✓ Raw results saved as 'architecture_results.csv'")
    
    # Analyze and visualize
    analyze_and_plot(results_df)
    
    # Generate report
    generate_report(results_df)
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    for f in sorted(generated_files):
        print(f"   - {f}")


if __name__ == "__main__":
    main()