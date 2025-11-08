import pandas as pd
import numpy as np

def load_concrete_data(path="concrete_data.csv", train_split=0.7):
    dataset = pd.read_csv(path)

    split_idx = int(dataset.shape[0] * train_split)
    training_dataset = dataset.iloc[:split_idx, :] 
    testing_dataset = dataset.iloc[split_idx:, :]

    X_train_raw = training_dataset.iloc[:, :8].values
    y_train_raw = training_dataset.iloc[:, 8].values.reshape(-1, 1)
    X_test_raw = testing_dataset.iloc[:, :8].values
    y_test_raw = testing_dataset.iloc[:, 8].values.reshape(-1, 1)

    X_mean = np.mean(X_train_raw, axis=0)
    X_std = np.std(X_train_raw, axis=0) + 1e-8 
    y_mean = np.mean(y_train_raw)
    y_std = np.std(y_train_raw) + 1e-8

    X_train = (X_train_raw - X_mean) / X_std
    y_train = (y_train_raw - y_mean) / y_std
    X_test = (X_test_raw - X_mean) / X_std
    y_test = (y_test_raw - y_mean) / y_std
    
    y_scale_params = (y_mean, y_std)
    
    return (X_train, y_train), (X_test, y_test), y_scale_params