"""
Load and test a previously saved optimized model
Run this AFTER training to verify the saved model works
"""

import numpy as np
import pickle
import ANN.ann as ann
import Utility.data_handler as data_handler
import Utility.model_utils as model_utils

def main():
    print("="*70)
    print("TESTING SAVED MODEL")
    print("="*70)
    
    # ============================================================
    # STEP 1: LOAD THE SAVED MODEL
    # ============================================================
    
    model_data = model_utils.load_optimized_model('model.pkl')
    
    # ============================================================
    # STEP 2: LOAD TEST DATA
    # ============================================================
    
    print("\nLoading test data...")
    (X_train, y_train), (X_test, y_test), y_scale_params = \
        data_handler.load_concrete_data("concrete_data.csv")
    
    print(f"Test set size: {len(X_test)} samples")
    
    # ============================================================
    # STEP 3: EVALUATE ON TEST SET
    # ============================================================
    
    metrics = model_utils.evaluate_saved_model(
        model_data=model_data,
        X_test=X_test,
        y_test=y_test,
        verbose=True
    )
    
    # ============================================================
    # STEP 4: VERIFY MODEL INFO
    # ============================================================
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Architecture: {model_data['layers']}")
    print(f"Total Parameters: {model_data['num_parameters']}")
    print(f"Training Iterations: {model_data['num_iterations']}")
    print(f"Loss Function: {model_data['loss_function']}")
    print(f"\nPSO Parameters:")
    for key, value in model_data['pso_params'].items():
        print(f"  {key}: {value}")
    
    # ============================================================
    # STEP 5: OPTIONAL - PLOT PREDICTIONS VS ACTUAL
    # ============================================================
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot 1: Predictions vs Actual
    ax1 = axes[0]
    ax1.scatter(metrics['actual_real'], metrics['predictions_real'], alpha=0.5)
    ax1.plot([metrics['actual_real'].min(), metrics['actual_real'].max()], 
             [metrics['actual_real'].min(), metrics['actual_real'].max()], 
             'r--', lw=2)
    ax1.set_xlabel('Actual Strength (MPa)')
    ax1.set_ylabel('Predicted Strength (MPa)')
    ax1.set_title(f"Predictions vs Actual (R²={metrics['R2']:.4f})")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    ax2.scatter(metrics['predictions_real'], metrics['residuals'], alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Strength (MPa)')
    ax2.set_ylabel('Residual (MPa)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n Evaluation plots saved as 'model_evaluation.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()