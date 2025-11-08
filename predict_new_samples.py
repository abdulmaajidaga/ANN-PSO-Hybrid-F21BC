"""
Use saved model to predict concrete strength for new samples
"""

import numpy as np
import pickle
import ANN.ann as ann
import Utility.model_utils as model_utils


def predict_concrete_strength(cement, blast_furnace_slag, fly_ash, water, 
                              superplasticizer, coarse_aggregate, fine_aggregate, age):
    """
    Predict concrete strength for a new concrete mix
    
    Args:
        cement: Cement content (kg/m³)
        blast_furnace_slag: Blast Furnace Slag content (kg/m³)
        fly_ash: Fly Ash content (kg/m³)
        water: Water content (kg/m³)
        superplasticizer: Superplasticizer content (kg/m³)
        coarse_aggregate: Coarse Aggregate content (kg/m³)
        fine_aggregate: Fine Aggregate content (kg/m³)
        age: Age of concrete (days)
        
    Returns:
        predicted_strength: Predicted compressive strength (MPa)
    """
    # Load model
    model_data = model_utils.load_optimized_model('best_concrete_model_100000.pkl')
    
    # Load training statistics for scaling
    with open('training_stats_100000.pkl', 'rb') as f:
        train_stats = pickle.load(f)
    
    # Create sample array
    sample = np.array([[cement, blast_furnace_slag, fly_ash, water, 
                       superplasticizer, coarse_aggregate, fine_aggregate, age]])
    
    # Make prediction
    prediction = model_utils.predict_new_sample(
        model_data=model_data,
        sample=sample,
        X_train_mean=train_stats['X_mean'],
        X_train_std=train_stats['X_std'],
        ann_module=ann
    )
    
    return prediction


def main():
    print("="*70)
    print("CONCRETE STRENGTH PREDICTOR")
    print("="*70)
    
    # Example 1: Standard concrete mix
    print("\nExample 1: Standard Concrete Mix")
    print("-"*70)
    strength = predict_concrete_strength(
        cement=540,
        blast_furnace_slag=0,
        fly_ash=0,
        water=162,
        superplasticizer=2.5,
        coarse_aggregate=1040,
        fine_aggregate=676,
        age=28
    )
    print(f"Predicted 28-day strength: {strength:.2f} MPa")
    
    # Example 2: High-performance concrete
    print("\nExample 2: High-Performance Concrete")
    print("-"*70)
    strength = predict_concrete_strength(
        cement=425,
        blast_furnace_slag=106.3,
        fly_ash=0,
        water=153.5,
        superplasticizer=16.5,
        coarse_aggregate=852.1,
        fine_aggregate=887.1,
        age=90
    )
    print(f"Predicted 90-day strength: {strength:.2f} MPa")
    
    # Example 3: Low cement content with fly ash
    print("\nExample 3: Eco-Friendly Mix (with Fly Ash)")
    print("-"*70)
    strength = predict_concrete_strength(
        cement=374.9,
        blast_furnace_slag=0,
        fly_ash=124.3,
        water=162.4,
        superplasticizer=6.0,
        coarse_aggregate=1049.9,
        fine_aggregate=780.0,
        age=56
    )
    print(f"Predicted 56-day strength: {strength:.2f} MPa")
    
    # Example 4: Predict strength at different ages
    print("\nExample 4: Strength Development Over Time")
    print("-"*70)
    mix_params = {
        'cement': 500,
        'blast_furnace_slag': 50,
        'fly_ash': 0,
        'water': 170,
        'superplasticizer': 5.0,
        'coarse_aggregate': 1000,
        'fine_aggregate': 750
    }
    
    ages = [3, 7, 14, 28, 56, 90, 180, 365]
    print(f"{'Age (days)':<15} {'Predicted Strength (MPa)':<25}")
    print("-"*70)
    for age in ages:
        strength = predict_concrete_strength(**mix_params, age=age)
        print(f"{age:<15} {strength:<25.2f}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()