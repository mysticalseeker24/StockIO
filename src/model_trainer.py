import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Import local module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_and_process_data, get_combined_data

def train_model(save_dir='../models'):
    """
    Train a Random Forest model on stock data from 2019-2021.
    
    Args:
        save_dir (str): Directory to save trained model and scaler
        
    Returns:
        tuple: Trained model and scaler objects
    """
    # Create models directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading and processing data...")
    # Load data from all stocks
    data_dict = load_and_process_data()
    
    # Get combined data for training period
    combined_data = get_combined_data(data_dict, 
                                     start_date='2019-01-01',
                                     end_date='2021-12-31')
    
    if combined_data.empty:
        raise ValueError("No data available for the specified date range")
    
    print(f"Combined data shape: {combined_data.shape}")
    
    # Define features for training
    features = ['RSI', 'volatility', 'momentum', 'trend_strength']
    
    # Prepare features and target
    X_train = combined_data[features]
    
    # Target: 1 if price goes up next day, -1 if it goes down
    y_train = (combined_data['Close'].shift(-1) > combined_data['Close']).astype(int) * 2 - 1
    
    # Drop last row since we don't have the next day's price
    X_train = X_train.iloc[:-1]
    y_train = y_train.iloc[:-1]
    
    print(f"Features shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Random Forest model...")
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    model_path = os.path.join(save_dir, 'random_forest_model.joblib')
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    
    print(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    
    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    return model, scaler

if __name__ == "__main__":
    train_model()
    print("Model training completed successfully.")
