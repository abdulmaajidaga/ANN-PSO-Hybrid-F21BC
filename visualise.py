import main as pso_ann_main
import matplotlib.pyplot as plt
import numpy as np

def run_and_visualise():
    """
    Runs the main PSO-ANN optimization and then generates
    visualizations for convergence and test set performance.
    """
    
    # 1. Run the main script to get the results
    print("Running main.py to get optimization data...")
    # This will run the entire optimization process defined in main.py
    loss_history, y_true, y_pred, loss_name = pso_ann_main.main()
    print("Optimization complete. Generating plots...")

    # 2. Create the plots
    plt.figure(figsize=(15, 6))

    # --- Plot 1: Loss Convergence Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('PSO Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel(f'Global Best Loss ({loss_name.upper()})')
    plt.grid(True)
    plt.tight_layout()

    # --- Plot 2: Actual vs. Predicted Scatter Plot ---
    plt.subplot(1, 2, 2)
    # Plot the actual vs. predicted values
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    
    # Plot the "perfect prediction" line (y=x)
    # Find the min/max values across both true and predicted for a nice line
    min_val = np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max([np.max(y_true), np.max(y_pred)])
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    
    plt.title('Test Set: Actual vs. Predicted Values')
    plt.xlabel('Actual Values (Un-scaled)')
    plt.ylabel('Predicted Values (Un-scaled)')
    plt.legend()
    plt.grid(True)
    # Make the axes equal to better visualize deviation from the y=x line
    plt.axis('equal') 
    
    # 3. Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure you have matplotlib installed: pip install matplotlib
    run_and_visualise()
