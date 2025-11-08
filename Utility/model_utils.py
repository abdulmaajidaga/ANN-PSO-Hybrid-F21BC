"""
Utility functions to save and load optimized MLP parameters
"""

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_optimized_model(optimizer, model_template, y_scale_params, pso_params, 
                        num_iterations, loss_function, filename='optimized_model.pkl'):
    """
    Save the best optimized model parameters to a file
    
    Args:
        optimizer: PSO optimizer with Gbest
        model_template: ANN model structure
        y_scale_params: (y_mean, y_std) tuple
        pso_params: Dictionary of PSO hyperparameters
        num_iterations: Number of PSO iterations
        loss_function: Loss function name used
        filename: Output filename
    """
    from BRIDGE import reconstruct_params
    
    y_mean, y_std = y_scale_params
    
    # Get best parameters
    best_params = reconstruct_params(optimizer.Gbest, model_template)
    
    # Package everything together
    model_data = {
        # Architecture
        'layers': model_template.layers,
        
        # Optimized parameters
        'weights': best_params[0],
        'biases': best_params[1],
        'activations': best_params[2],
        
        # Training info
        'best_loss': optimizer.Gbest_value,
        'loss_history': optimizer.Gbest_value_history,
        'mean_fitness_history': optimizer.mean_fitness_history,
        
        # Scaling info (needed for predictions)
        'y_mean': y_mean,
        'y_std': y_std,
        
        # Configuration
        'pso_params': pso_params,
        'num_iterations': num_iterations,
        'loss_function': loss_function,
        
        # Metadata
        'num_parameters': sum(w.size + b.size for w, b in zip(best_params[0], best_params[1]))
    }
    
    # Save using pickle
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n{'='*60}")
    print(f"MODEL SAVED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Filename: {filename}")
    print(f"Architecture: {model_data['layers']}")
    print(f"Total Parameters: {model_data['num_parameters']}")
    print(f"Best Training Loss: {model_data['best_loss']:.6f}")
    print(f"Loss Function: {loss_function}")
    print(f"{'='*60}")
    
    return model_data


def load_optimized_model(filename='optimized_model.pkl'):
    """
    Load a previously saved optimized model
    
    Args:
        filename: Input filename
        
    Returns:
        model_data: Dictionary with weights, biases, etc.
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"MODEL LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Filename: {filename}")
    print(f"Architecture: {model_data['layers']}")
    print(f"Total Parameters: {model_data['num_parameters']}")
    print(f"Best Training Loss: {model_data['best_loss']:.6f}")
    print(f"Loss Function: {model_data['loss_function']}")
    print(f"{'='*60}")
    
    return model_data


def predict_with_saved_model(model_data, X, ann_module):
    """
    Make predictions using saved model parameters
    
    Args:
        model_data: Dictionary from load_optimized_model()
        X: Input data (MUST be scaled the same way as training data)
        ann_module: Your ANN module (import ANN.ann as ann)
        
    Returns:
        predictions: Model predictions (scaled - need to unscale for real values)
    """
    # Create model with same architecture
    model = ann_module.MultiLayerANN(layers=model_data['layers'])
    
    # Use the saved parameters
    params = [model_data['weights'], model_data['biases'], model_data['activations']]
    
    # Make predictions
    predictions = model.evaluate_with_params(X, params=params)
    
    return predictions


def evaluate_saved_model(model_data, X_test, y_test, ann_module, verbose=True):
    """
    Full evaluation of saved model on test set
    
    Args:
        model_data: Dictionary from load_optimized_model()
        X_test: Test features (MUST be scaled)
        y_test: Test labels (MUST be scaled)
        ann_module: Your ANN module
        verbose: Print results
        
    Returns:
        metrics: Dictionary with all metrics
    """
    y_mean = model_data['y_mean']
    y_std = model_data['y_std']
    
    # Get predictions (scaled)
    predictions_scaled = predict_with_saved_model(model_data, X_test, ann_module)
    
    # Unscale to original values
    predictions_real = (predictions_scaled * y_std) + y_mean
    y_test_real = (y_test * y_std) + y_mean
    
    # Calculate metrics
    mse = mean_squared_error(y_test_real, predictions_real)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)
    mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
    
    # Calculate residuals
    residuals = y_test_real - predictions_real
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'predictions_real': predictions_real,
        'actual_real': y_test_real,
        'residuals': residuals,
        'predictions_scaled': predictions_scaled
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} MPa")
        print(f"Mean Absolute Error (MAE):      {mae:.4f} MPa")
        print(f"Mean Squared Error (MSE):       {mse:.4f}")
        print(f"R² Score:                       {r2:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print("="*60)
        
        # Show sample predictions
        print("\nSAMPLE PREDICTIONS (First 10 samples)")
        print("-"*60)
        print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12} {'% Error':<10}")
        print("-"*60)
        
        for i in range(min(10, len(y_test_real))):
            actual = y_test_real[i, 0]
            predicted = predictions_real[i, 0]
            error = actual - predicted
            pct_error = (error / actual) * 100
            print(f"{actual:<12.2f} {predicted:<12.2f} {error:<12.2f} {pct_error:<10.2f}%")
        print("-"*60)
    
    return metrics


def predict_new_sample(model_data, sample, X_train_mean, X_train_std, ann_module):
    """
    Predict concrete strength for a single new sample
    
    Args:
        model_data: Dictionary from load_optimized_model()
        sample: Array of features [cement, slag, ash, water, superplastic, coarse_agg, fine_agg, age]
        X_train_mean: Mean values from training data (for scaling)
        X_train_std: Std values from training data (for scaling)
        ann_module: Your ANN module
        
    Returns:
        prediction: Predicted concrete strength in MPa
    """
    # Ensure sample is 2D array
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    
    # Scale the input using training statistics
    sample_scaled = (sample - X_train_mean) / X_train_std
    
    # Get prediction (scaled)
    pred_scaled = predict_with_saved_model(model_data, sample_scaled, ann_module)
    
    # Unscale to real value
    pred_real = (pred_scaled * model_data['y_std']) + model_data['y_mean']
    
    return pred_real[0, 0]