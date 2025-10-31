# data_loader.py
import pandas as pd
import numpy as np

def load_concrete_data(path="concrete_data.csv", train_split=0.7):
    """
    Loads and preprocesses the concrete dataset.
    
    This function reads a CSV, splits it into training and testing sets,
    and then applies manual standardization (z-score normalization) to
    the features (X) and target (y).
    
    Standardization is done using *only* the mean and std dev of the 
    training data to prevent data leakage from the test set.
    """
    # Read the dataset from the CSV file
    dataset = pd.read_csv(path)

    # Calculate the index at which to split the data
    split_idx = int(dataset.shape[0] * train_split)
    # Slice the DataFrame for training
    training_dataset = dataset.iloc[:split_idx, :] 
    # Slice the DataFrame for testing
    testing_dataset = dataset.iloc[split_idx:, :]

    # --- NEW: Manual Standardization ---
    
    # 1. Get features (X) and targets (y) as numpy arrays
    # Features are the first 8 columns
    X_train_raw = training_dataset.iloc[:, :8].values
    # Target is the 9th column, reshaped to be a column vector
    y_train_raw = training_dataset.iloc[:, 8].values.reshape(-1, 1)

    X_test_raw = testing_dataset.iloc[:, :8].values
    y_test_raw = testing_dataset.iloc[:, 8].values.reshape(-1, 1)

    # 2. Calculate mean and std DEV *from training data only*
    #    axis=0 calculates stats for each column (feature) independently
    X_mean = np.mean(X_train_raw, axis=0)
    # Add 1e-8 (a small epsilon) to std dev to prevent division by zero
    # if a feature column has zero variance.
    X_std = np.std(X_train_raw, axis=0) + 1e-8 
    
    y_mean = np.mean(y_train_raw)
    y_std = np.std(y_train_raw) + 1e-8

    # 3. Apply the scaling (z-score = (value - mean) / std_dev)
    X_train = (X_train_raw - X_mean) / X_std
    y_train = (y_train_raw - y_mean) / y_std
    
    # 4. Apply the *same* training stats to the test data
    # This is crucial to treat the test data as unseen.
    X_test = (X_test_raw - X_mean) / X_std
    y_test = (y_test_raw - y_mean) / y_std
    
    # 5. Return scaled data and the scaling params for 'y'
    # The 'y' params are needed later to un-scale predictions.
    y_scale_params = (y_mean, y_std)
    
    return (X_train, y_train), (X_test, y_test), y_scale_params