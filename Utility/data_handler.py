import pandas as pd
import numpy as np


class DataHandler(object):
    def  __init__(self):
        print("Initialized Data Handler")
        self.X_mean = None
        self.X_std = None

        
    def transform_data(self, path="concrete_data.csv", train_split=0.7, random_seed = 1, remove_outliers=False):
        # --- 1️⃣ Load dataset ---
        dataset = pd.read_csv(path)
        dataset.columns = dataset.columns.str.strip()  # clean column names

        # --- 2️⃣ Shuffle dataset to avoid distribution drift ---
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # --- 3️⃣ Handle missing or infinite values ---
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(inplace=True)

        # --- 4️⃣ Optional: Remove outliers using IQR method ---
        if remove_outliers:
            feature_cols = dataset.columns[:8]
            Q1 = dataset[feature_cols].quantile(0.25)
            Q3 = dataset[feature_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataset = dataset[~((dataset[feature_cols] < lower_bound) | (dataset[feature_cols] > upper_bound)).any(axis=1)]

        # --- 5️⃣ Split into training and testing ---
        split_idx = int(dataset.shape[0] * train_split)
        training_dataset = dataset.iloc[:split_idx, :]
        testing_dataset = dataset.iloc[split_idx:, :]

        X_train_raw = training_dataset.iloc[:, :8].values
        y_train_raw = training_dataset.iloc[:, 8].values.reshape(-1, 1)
        X_test_raw = testing_dataset.iloc[:, :8].values
        y_test_raw = testing_dataset.iloc[:, 8].values.reshape(-1, 1)

        self.X_mean = np.mean(X_train_raw, axis=0)
        self.X_std = np.std(X_train_raw, axis=0) + 1e-8 

        X_train_scaled = (X_train_raw -  self.X_mean) /  self.X_std
        X_test_scaled = (X_test_raw -  self.X_mean) /  self.X_std 
        
        return (X_train_scaled, y_train_raw), (X_test_scaled, y_test_raw)
    
    def inverse_transform_data(self, X_predictions_scaled):
        X_predictions_real = X_predictions_scaled * self.X_std + self.X_mean
        return X_predictions_real
        

