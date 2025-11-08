"""
Complete ANN Architecture Investigation
Single file that runs experiments and generates analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Your existing imports
import Utility.data_loader as data_loader
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
# PSO_PARAMS = {
#     'alpha': 0.8,   
#     'beta': 1.2,    
#     'gamma': 1.5,     
#     'delta': 0.5,   
#     'epsilon': 0.8  
# }

# PSO configuration
PSO_CONFIG = {
    'num_particles': 50,        # Adjust based on your computational budget
    'num_iterations': 500,      # Adjust based on your computational budget
    'num_informants': 10,
    'loss_function': 'rmse'
}

# Experiment settings
NUM_RUNS = 3  # How many times to repeat each configuration
DATASET_PATH = "concrete_data.csv"


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
            real_loss = scaled_loss * y_std if PSO_CONFIG['loss_function'] in ['rmse', 'mae'] else scaled_loss * (y_std ** 2)
            print(f"  Iter {i+1}: Loss = {real_loss:.6f}")
    
    # Get results
    initial_loss_scaled = optimizer.Gbest_value_history[0]
    final_loss_scaled = optimizer.Gbest_value_history[-1]
    
    # Unscale losses
    if PSO_CONFIG['loss_function'] == 'mse':
        initial_loss = initial_loss_scaled * (y_std ** 2)
        final_loss = final_loss_scaled * (y_std ** 2)
    elif PSO_CONFIG['loss_function'] in ['rmse', 'mae']:
        initial_loss = initial_loss_scaled * y_std
        final_loss = final_loss_scaled * y_std
    else:
        initial_loss = initial_loss_scaled
        final_loss = final_loss_scaled
    
    # Test set evaluation
    best_params = reconstruct_params(optimizer.Gbest, model)
    test_predictions_scaled = model.evaluate_with_params(X_test, best_params)
    test_predictions_real = (test_predictions_scaled * y_std) + y_mean
    y_test_real = (y_test * y_std) + y_mean
    
    test_loss_func = loss_functions.get_loss_function(PSO_CONFIG['loss_function'])
    test_loss = test_loss_func(y_test_real, test_predictions_real)
    
    # Count parameters
    total_params = sum(architecture['layers'][i] * architecture['layers'][i+1] + architecture['layers'][i+1] 
                      for i in range(len(architecture['layers']) - 1))
    
    # Find convergence iteration (when reaches 95% of final improvement)
    convergence_iteration = 0
    threshold = initial_loss_scaled - (initial_loss_scaled - final_loss_scaled) * 0.95
    for i, loss in enumerate(optimizer.Gbest_value_history):
        if loss <= threshold:
            convergence_iteration = i
            break
    
    return {
        'run_id': run_id,
        'layers': str(architecture['layers']),
        'num_layers': len(architecture['layers']),
        'total_params': total_params,
        'initial_train_loss': initial_loss,
        'final_train_loss': final_loss,
        'test_loss': test_loss,
        'improvement_percent': ((initial_loss - final_loss) / initial_loss) * 100,
        'convergence_iteration': convergence_iteration,
        'generalization_gap': test_loss - final_loss,
    }


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_depth(X_train, y_train, X_test, y_test, y_scale_params, results):
    """Experiment 1: Effect of network depth"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Effect of Network Depth")
    print("="*70)
    
    base_width = 16
    depths = [2, 3, 4, 5]
    
    for depth in depths:
        if depth == 2:
            layers = [8, 1]  # No hidden layers
        else:
            layers = [8] + [base_width] * (depth - 2) + [1]
        
        architecture = {'layers': layers}
        
        for run in range(NUM_RUNS):
            result = run_single_experiment(architecture, X_train, y_train, X_test, y_test, y_scale_params, run)
            result['experiment'] = 'depth'
            result['depth'] = depth
            results.append(result)


def experiment_width(X_train, y_train, X_test, y_test, y_scale_params, results):
    """Experiment 2: Effect of network width"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Effect of Network Width")
    print("="*70)
    
    num_hidden_layers = 2
    widths = [4, 8, 16, 32, 64]
    
    for width in widths:
        layers = [8] + [width] * num_hidden_layers + [1]
        architecture = {'layers': layers}
        
        for run in range(NUM_RUNS):
            result = run_single_experiment(architecture, X_train, y_train, X_test, y_test, y_scale_params, run)
            result['experiment'] = 'width'
            result['width'] = width
            results.append(result)


def experiment_shapes(X_train, y_train, X_test, y_test, y_scale_params, results):
    """Experiment 3: Effect of architecture shape"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Effect of Architecture Shape")
    print("="*70)
    
    architectures = [
        {'name': 'pyramid_expanding', 'layers': [8, 16, 32, 64, 1]},
        {'name': 'pyramid_contracting', 'layers': [8, 64, 32, 16, 1]},
        {'name': 'diamond', 'layers': [8, 16, 32, 16, 1]},
        {'name': 'rectangle', 'layers': [8, 16, 16, 16, 1]},
        {'name': 'bottleneck', 'layers': [8, 32, 8, 32, 1]},
    ]
    
    for arch in architectures:
        architecture = {'layers': arch['layers']}
        
        for run in range(NUM_RUNS):
            result = run_single_experiment(architecture, X_train, y_train, X_test, y_test, y_scale_params, run)
            result['experiment'] = 'shape'
            result['shape'] = arch['name']
            results.append(result)


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_and_plot(results_df):
    """Analyze results and create visualizations"""
    
    print("\n" + "="*70)
    print("ANALYSIS AND VISUALIZATION")
    print("="*70)
    
    sns.set_style("whitegrid")
    
    # ---- DEPTH ANALYSIS ----
    if 'depth' in results_df['experiment'].values:
        print("\n--- Depth Analysis ---")
        depth_df = results_df[results_df['experiment'] == 'depth']
        
        summary = depth_df.groupby('depth').agg({
            'final_train_loss': ['mean', 'std'],
            'test_loss': ['mean', 'std'],
            'convergence_iteration': 'mean',
            'total_params': 'first'
        }).round(4)
        print(summary)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Effect of Network Depth', fontsize=16, fontweight='bold')
        
        # Plot 1: Training vs Test Loss
        ax1 = axes[0, 0]
        depth_summary = depth_df.groupby('depth').agg({
            'final_train_loss': 'mean',
            'test_loss': 'mean'
        })
        depth_summary.plot(kind='bar', ax=ax1)
        ax1.set_title('Training vs Test Loss by Depth')
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('Loss (RMSE)')
        ax1.legend(['Training Loss', 'Test Loss'])
        ax1.tick_params(axis='x', rotation=0)
        
        # Plot 2: Box plot
        ax2 = axes[0, 1]
        depth_df.boxplot(column='test_loss', by='depth', ax=ax2)
        ax2.set_title('Test Loss Distribution by Depth')
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('Test Loss (RMSE)')
        plt.sca(ax2)
        plt.xticks(rotation=0)
        
        # Plot 3: Convergence speed
        ax3 = axes[1, 0]
        conv_summary = depth_df.groupby('depth')['convergence_iteration'].mean()
        conv_summary.plot(kind='bar', ax=ax3, color='coral')
        ax3.set_title('Convergence Speed by Depth')
        ax3.set_xlabel('Number of Layers')
        ax3.set_ylabel('Iterations to 95% Convergence')
        ax3.tick_params(axis='x', rotation=0)
        
        # Plot 4: Parameters vs Performance
        ax4 = axes[1, 1]
        param_perf = depth_df.groupby('depth').agg({
            'total_params': 'first',
            'test_loss': 'mean'
        })
        ax4.scatter(param_perf['total_params'], param_perf['test_loss'], s=100)
        for idx, row in param_perf.iterrows():
            ax4.annotate(f'{idx} layers', 
                        (row['total_params'], row['test_loss']),
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Total Parameters')
        ax4.set_ylabel('Mean Test Loss')
        ax4.set_title('Model Complexity vs Performance')
        
        plt.tight_layout()
        plt.savefig('depth_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Depth analysis saved as 'depth_analysis.png'")
        plt.show()
    
    # ---- WIDTH ANALYSIS ----
    if 'width' in results_df['experiment'].values:
        print("\n--- Width Analysis ---")
        width_df = results_df[results_df['experiment'] == 'width']
        
        summary = width_df.groupby('width').agg({
            'final_train_loss': ['mean', 'std'],
            'test_loss': ['mean', 'std'],
            'total_params': 'first'
        }).round(4)
        print(summary)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Effect of Network Width', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance by width
        ax1 = axes[0, 0]
        width_summary = width_df.groupby('width').agg({
            'final_train_loss': 'mean',
            'test_loss': 'mean'
        })
        width_summary.plot(kind='line', marker='o', ax=ax1)
        ax1.set_title('Performance by Network Width')
        ax1.set_xlabel('Neurons per Hidden Layer')
        ax1.set_ylabel('Loss (RMSE)')
        ax1.legend(['Training Loss', 'Test Loss'])
        ax1.grid(True)
        
        # Plot 2: Generalization gap
        ax2 = axes[0, 1]
        width_df['gen_gap'] = width_df['test_loss'] - width_df['final_train_loss']
        gap_summary = width_df.groupby('width')['gen_gap'].mean()
        gap_summary.plot(kind='bar', ax=ax2, color='salmon')
        ax2.set_title('Generalization Gap by Width')
        ax2.set_xlabel('Neurons per Hidden Layer')
        ax2.set_ylabel('Test Loss - Train Loss')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax2.tick_params(axis='x', rotation=0)
        
        # Plot 3: Violin plot
        ax3 = axes[1, 0]
        sns.violinplot(data=width_df, x='width', y='test_loss', ax=ax3)
        ax3.set_title('Test Loss Distribution by Width')
        ax3.set_xlabel('Neurons per Hidden Layer')
        ax3.set_ylabel('Test Loss (RMSE)')
        
        # Plot 4: Parameters vs performance
        ax4 = axes[1, 1]
        param_summary = width_df.groupby('width').agg({
            'total_params': 'first',
            'test_loss': ['mean', 'std']
        })
        ax4.errorbar(param_summary['total_params']['first'], 
                    param_summary['test_loss']['mean'],
                    yerr=param_summary['test_loss']['std'],
                    marker='o', capsize=5, linewidth=2)
        ax4.set_xlabel('Total Parameters')
        ax4.set_ylabel('Mean Test Loss')
        ax4.set_title('Complexity vs Performance Trade-off')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('width_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Width analysis saved as 'width_analysis.png'")
        plt.show()
    
    # ---- SHAPE ANALYSIS ----
    if 'shape' in results_df['experiment'].values:
        print("\n--- Shape Analysis ---")
        shape_df = results_df[results_df['experiment'] == 'shape']
        
        summary = shape_df.groupby('shape').agg({
            'test_loss': ['mean', 'std'],
            'convergence_iteration': 'mean',
            'layers': 'first'
        }).round(4)
        print(summary)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Effect of Architecture Shape', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance comparison
        ax1 = axes[0]
        shape_summary = shape_df.groupby('shape')['test_loss'].agg(['mean', 'std'])
        shape_summary['mean'].plot(kind='barh', xerr=shape_summary['std'], 
                                   ax=ax1, color='skyblue')
        ax1.set_title('Performance by Architecture Shape')
        ax1.set_xlabel('Test Loss (RMSE)')
        ax1.set_ylabel('Architecture Shape')
        
        # Plot 2: Convergence comparison
        ax2 = axes[1]
        conv_summary = shape_df.groupby('shape')['convergence_iteration'].mean()
        conv_summary.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Convergence Speed by Shape')
        ax2.set_xlabel('Iterations to 95% Convergence')
        ax2.set_ylabel('Architecture Shape')
        
        plt.tight_layout()
        plt.savefig('shape_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Shape analysis saved as 'shape_analysis.png'")
        plt.show()


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
        top5 = results_df.nsmallest(5, 'test_loss')[['layers', 'test_loss', 
                                                      'total_params', 'convergence_iteration']]
        f.write(top5.to_string(index=False) + "\n\n")
        
        # Findings per experiment
        for exp in results_df['experiment'].unique():
            f.write(f"\n{exp.upper()} EXPERIMENT\n")
            f.write("-"*70 + "\n")
            exp_df = results_df[results_df['experiment'] == exp]
            f.write(exp_df.describe().to_string() + "\n\n")
    
    print("\n✓ Report saved as 'architecture_report.txt'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("ANN ARCHITECTURE INVESTIGATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Particles: {PSO_CONFIG['num_particles']}")
    print(f"  Iterations: {PSO_CONFIG['num_iterations']}")
    print(f"  Runs per config: {NUM_RUNS}")
    print(f"  Loss function: {PSO_CONFIG['loss_function']}")
    
    # Load data
    print("\nLoading data...")
    (X_train, y_train), (X_test, y_test), y_scale_params = \
        data_loader.load_concrete_data(DATASET_PATH)
    
    # Run experiments
    results = []
    
    # Choose which experiments to run (comment out if you don't want to run)
    experiment_depth(X_train, y_train, X_test, y_test, y_scale_params, results)
    experiment_width(X_train, y_train, X_test, y_test, y_scale_params, results)
    experiment_shapes(X_train, y_train, X_test, y_test, y_scale_params, results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
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
    print("  - architecture_results.csv (raw data)")
    print("  - architecture_report.txt (summary)")
    print("  - depth_analysis.png (if depth experiment was run)")
    print("  - width_analysis.png (if width experiment was run)")
    print("  - shape_analysis.png (if shape experiment was run)")


if __name__ == "__main__":
    main()