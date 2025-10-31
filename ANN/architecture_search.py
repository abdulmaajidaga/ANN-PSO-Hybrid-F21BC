"""
Automated Neural Architecture Search for Concrete Strength Prediction
Tests multiple layer configurations and activation functions
"""
import numpy as np
import itertools
from ANN.ann import MultiLayerANN
from ANN.loss_functions import mean_squared_error, root_mean_squared_error
from Utility.data_loader import load_concrete_data
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

class ArchitectureSearch:
    """
    Automated search for best ANN architecture and activation functions.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, y_scale_params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_mean, self.y_std = y_scale_params
        
        self.results = []
        
    def generate_architectures(self, max_layers=4, min_neurons=4, max_neurons=32):
        """
        Generate different layer architectures to test.
        
        Args:
            max_layers: Maximum number of hidden layers (1-4)
            min_neurons: Minimum neurons per hidden layer
            max_neurons: Maximum neurons per hidden layer
        
        Returns:
            List of architecture configurations [8, h1, h2, ..., 1]
        """
        architectures = []
        neuron_options = [4, 8, 12, 16, 20, 24, 32]
        neuron_options = [n for n in neuron_options if min_neurons <= n <= max_neurons]
        
        # 1 hidden layer: [8, n1, 1]
        for n1 in neuron_options:
            architectures.append([8, n1, 1])
        
        if max_layers >= 2:
            # 2 hidden layers: [8, n1, n2, 1]
            for n1 in neuron_options:
                for n2 in neuron_options:
                    if n2 <= n1:  # Generally decrease size
                        architectures.append([8, n1, n2, 1])
        
        if max_layers >= 3:
            # 3 hidden layers: [8, n1, n2, n3, 1]
            for n1 in [16, 20, 24, 32]:
                for n2 in [8, 12, 16]:
                    for n3 in [4, 8]:
                        if n2 <= n1 and n3 <= n2:
                            architectures.append([8, n1, n2, n3, 1])
        
        if max_layers >= 4:
            # 4 hidden layers: [8, n1, n2, n3, n4, 1]
            for n1 in [24, 32]:
                for n2 in [16, 20]:
                    for n3 in [12, 16]:
                        for n4 in [8, 12]:
                            if n2 <= n1 and n3 <= n2 and n4 <= n3:
                                architectures.append([8, n1, n2, n3, n4, 1])
        
        return architectures
    
    def generate_activation_combinations(self, num_hidden_layers):
        """
        Generate all possible activation function combinations for hidden layers.
        
        Args:
            num_hidden_layers: Number of hidden layers
        
        Returns:
            List of activation combinations
        """
        available_activations = ['logistic', 'relu', 'tanh', 'linear']
        
        if num_hidden_layers == 1:
            # For single hidden layer, test each activation
            return [[act] for act in available_activations]
        
        elif num_hidden_layers == 2:
            # For 2 layers, test common combinations
            combinations = []
            for act1 in available_activations:
                for act2 in available_activations:
                    combinations.append([act1, act2])
            return combinations
        
        else:
            # For 3+ layers, test selected combinations to avoid explosion
            combinations = [
                ['relu'] * num_hidden_layers,
                ['tanh'] * num_hidden_layers,
                ['logistic'] * num_hidden_layers,
                ['relu', 'relu', 'tanh'] + ['tanh'] * (num_hidden_layers - 3),
                ['relu', 'tanh', 'relu'] + ['relu'] * (num_hidden_layers - 3),
                ['tanh', 'relu', 'relu'] + ['relu'] * (num_hidden_layers - 3),
            ]
            return [c[:num_hidden_layers] for c in combinations]
    
    def train_simple_gd(self, model, activations, epochs=500, lr=0.001):
        """
        Simple gradient descent training (basic version).
        Note: For production, use PSO optimization instead.
        """
        best_test_mse = float('inf')
        
        for epoch in range(epochs):
            # Forward pass
            predictions = model.compute_forward(self.X_train, model.weights, 
                                               model.biases, activations)
            
            # Calculate error
            error = predictions - self.y_train
            
            # Simple weight updates (very basic backprop approximation)
            for i in range(len(model.weights)):
                if i == 0:
                    grad_w = np.dot(self.X_train.T, error) / len(self.X_train)
                else:
                    grad_w = np.dot(predictions.T, error) / len(predictions)
                
                grad_b = np.mean(error, axis=0, keepdims=True)
                
                model.weights[i] -= lr * grad_w * 0.01
                model.biases[i] -= lr * grad_b * 0.01
            
            # Evaluate on test set
            if epoch % 100 == 0:
                test_pred = model.compute_forward(self.X_test, model.weights, 
                                                  model.biases, activations)
                test_mse = mean_squared_error(self.y_test, test_pred)
                best_test_mse = min(best_test_mse, test_mse)
        
        return best_test_mse
    
    def evaluate_architecture(self, architecture, activations, training_iterations=500):
        """
        Evaluate a specific architecture with given activations.
        
        Returns:
            Dictionary with results
        """
        num_hidden = len(architecture) - 2
        
        # Create model
        model = MultiLayerANN(architecture, activations)
        
        # Initial random predictions
        train_pred_init = model.predict(self.X_train)
        test_pred_init = model.predict(self.X_test)
        
        init_train_mse = mean_squared_error(self.y_train, train_pred_init)
        init_test_mse = mean_squared_error(self.y_test, test_pred_init)
        
        # Train with simple gradient descent
        # NOTE: Replace this with PSO for better results!
        trained_test_mse = self.train_simple_gd(model, activations, 
                                                epochs=training_iterations)
        
        # Final evaluation
        train_pred_final = model.predict(self.X_train)
        test_pred_final = model.predict(self.X_test)
        
        final_train_mse = mean_squared_error(self.y_train, train_pred_final)
        final_test_mse = mean_squared_error(self.y_test, test_pred_final)
        
        # Unscaled metrics
        test_pred_unscaled = (test_pred_final * self.y_std) + self.y_mean
        y_test_unscaled = (self.y_test * self.y_std) + self.y_mean
        final_test_rmse_unscaled = root_mean_squared_error(y_test_unscaled, 
                                                           test_pred_unscaled)
        
        # Count parameters
        total_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
        
        return {
            'architecture': architecture,
            'activations': activations,
            'num_hidden_layers': num_hidden,
            'total_params': total_params,
            'init_train_mse': init_train_mse,
            'init_test_mse': init_test_mse,
            'final_train_mse': final_train_mse,
            'final_test_mse': final_test_mse,
            'final_test_rmse_unscaled': final_test_rmse_unscaled,
        }
    
    def search(self, max_layers=3, max_neurons=32, training_iterations=500, 
               test_activations=True):
        """
        Perform comprehensive architecture search.
        
        Args:
            max_layers: Maximum number of hidden layers to test
            max_neurons: Maximum neurons per layer
            training_iterations: Epochs for simple GD training
            test_activations: If True, test different activation combinations
        """
        print("="*70)
        print("AUTOMATED NEURAL ARCHITECTURE SEARCH")
        print("="*70)
        print(f"Search space:")
        print(f"  - Max hidden layers: {max_layers}")
        print(f"  - Max neurons per layer: {max_neurons}")
        print(f"  - Training iterations: {training_iterations}")
        print(f"  - Test activation combinations: {test_activations}")
        print("="*70)
        
        # Generate architectures
        architectures = self.generate_architectures(max_layers, 4, max_neurons)
        print(f"\nGenerated {len(architectures)} architectures to test")
        
        total_tests = 0
        for arch in architectures:
            num_hidden = len(arch) - 2
            if test_activations:
                act_combos = self.generate_activation_combinations(num_hidden)
                total_tests += len(act_combos)
            else:
                total_tests += 1
        
        print(f"Total configurations to test: {total_tests}")
        print("\nStarting search...\n")
        
        # Test each architecture
        test_count = 0
        for arch_idx, architecture in enumerate(architectures):
            num_hidden = len(architecture) - 2
            
            if test_activations:
                activation_combos = self.generate_activation_combinations(num_hidden)
            else:
                activation_combos = [['relu'] * num_hidden]
            
            for act_combo in activation_combos:
                test_count += 1
                
                print(f"[{test_count}/{total_tests}] Testing: {architecture} "
                      f"with activations {act_combo}")
                
                result = self.evaluate_architecture(architecture, act_combo, 
                                                    training_iterations)
                self.results.append(result)
                
                print(f"  → Test RMSE (unscaled): {result['final_test_rmse_unscaled']:.4f} MPa")
                print(f"  → Test MSE (scaled): {result['final_test_mse']:.6f}")
                print(f"  → Parameters: {result['total_params']}")
                print()
        
        print("\n" + "="*70)
        print("SEARCH COMPLETE!")
        print("="*70)
    
    def get_best_results(self, top_n=10, sort_by='final_test_rmse_unscaled'):
        """
        Get the top N best architectures.
        
        Args:
            top_n: Number of top results to return
            sort_by: Metric to sort by
        
        Returns:
            List of top N results
        """
        sorted_results = sorted(self.results, key=lambda x: x[sort_by])
        return sorted_results[:top_n]
    
    def print_summary(self, top_n=10):
        """Print summary of top architectures."""
        best_results = self.get_best_results(top_n)
        
        print("\n" + "="*70)
        print(f"TOP {top_n} ARCHITECTURES")
        print("="*70)
        
        for i, result in enumerate(best_results, 1):
            print(f"\n#{i}")
            print(f"  Architecture: {result['architecture']}")
            print(f"  Activations: {result['activations']}")
            print(f"  Parameters: {result['total_params']}")
            print(f"  Test RMSE (unscaled): {result['final_test_rmse_unscaled']:.4f} MPa")
            print(f"  Test MSE (scaled): {result['final_test_mse']:.6f}")
            print(f"  Train MSE (scaled): {result['final_train_mse']:.6f}")
    
    def save_results(self, filename='architecture_search_results.json'):
        """Save all results to JSON file."""
        # Convert numpy types to native Python types
        results_serializable = []
        for r in self.results:
            r_copy = r.copy()
            r_copy['architecture'] = [int(x) for x in r_copy['architecture']]
            for key in ['init_train_mse', 'init_test_mse', 'final_train_mse', 
                       'final_test_mse', 'final_test_rmse_unscaled']:
                r_copy[key] = float(r_copy[key])
            results_serializable.append(r_copy)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def save_csv(self, filename='architecture_search_results.csv'):
        """Save results to CSV file."""
        df = pd.DataFrame(self.results)
        df['architecture'] = df['architecture'].apply(str)
        df['activations'] = df['activations'].apply(str)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def plot_results(self, filename='architecture_comparison.png'):
        """Create visualization of results."""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sort by test RMSE
        df_sorted = df.sort_values('final_test_rmse_unscaled')
        top_20 = df_sorted.head(20)
        
        # Plot 1: Top 20 architectures by test RMSE
        ax1 = axes[0, 0]
        arch_labels = [f"{r['architecture']}\n{r['activations']}" 
                      for _, r in top_20.iterrows()]
        x_pos = range(len(top_20))
        ax1.barh(x_pos, top_20['final_test_rmse_unscaled'])
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(arch_labels, fontsize=6)
        ax1.set_xlabel('Test RMSE (MPa)')
        ax1.set_title('Top 20 Architectures by Test RMSE')
        ax1.invert_yaxis()
        
        # Plot 2: Parameters vs Performance
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['total_params'], df['final_test_rmse_unscaled'],
                             c=df['num_hidden_layers'], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('Total Parameters')
        ax2.set_ylabel('Test RMSE (MPa)')
        ax2.set_title('Model Complexity vs Performance')
        plt.colorbar(scatter, ax=ax2, label='Hidden Layers')
        
        # Plot 3: Hidden layers vs Performance
        ax3 = axes[1, 0]
        grouped = df.groupby('num_hidden_layers')['final_test_rmse_unscaled']
        means = grouped.mean()
        stds = grouped.std()
        ax3.bar(means.index, means.values, yerr=stds.values, alpha=0.7, capsize=5)
        ax3.set_xlabel('Number of Hidden Layers')
        ax3.set_ylabel('Test RMSE (MPa)')
        ax3.set_title('Performance by Number of Hidden Layers')
        ax3.set_xticks(means.index)
        
        # Plot 4: Train vs Test MSE (overfitting check)
        ax4 = axes[1, 1]
        ax4.scatter(df['final_train_mse'], df['final_test_mse'], alpha=0.5)
        ax4.plot([df['final_train_mse'].min(), df['final_train_mse'].max()],
                [df['final_train_mse'].min(), df['final_train_mse'].max()],
                'r--', label='Perfect fit')
        ax4.set_xlabel('Train MSE (scaled)')
        ax4.set_ylabel('Test MSE (scaled)')
        ax4.set_title('Train vs Test MSE (Overfitting Check)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        plt.close()


def main():
    """Main execution function."""
    print("Loading concrete strength data...")
    (X_train, y_train), (X_test, y_test), y_scale_params = load_concrete_data(
        path="concrete_data.csv"
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Input features: {X_train.shape[1]}")
    
    # Create architecture search
    searcher = ArchitectureSearch(X_train, y_train, X_test, y_test, y_scale_params)
    
    # Perform search
    # Adjust these parameters based on how long you want to wait:
    # - max_layers: 2 (fast), 3 (medium), 4 (slow)
    # - test_activations: False (fast), True (thorough)
    # - training_iterations: 300 (fast), 500 (medium), 1000 (slow)
    
    searcher.search(
        max_layers=3,              # Test up to 3 hidden layers
        max_neurons=32,            # Max 32 neurons per layer
        training_iterations=500,   # 500 epochs of training
        test_activations=True      # Test different activation combos
    )
    
    # Print summary
    searcher.print_summary(top_n=15)
    
    # Save results
    searcher.save_results('architecture_search_results.json')
    searcher.save_csv('architecture_search_results.csv')
    
    # Create visualizations
    searcher.plot_results('architecture_comparison.png')
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    best = searcher.get_best_results(top_n=1)[0]
    print(f"Best architecture: {best['architecture']}")
    print(f"Best activations: {best['activations']}")
    print(f"Use this for your PSO optimization!")
    print("="*70)


if __name__ == '__main__':
    main()