import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import joblib

# Add the parent directory to the path to import the data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_combined_data, load_and_process_data

def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM training
    
    Args:
        data (DataFrame): DataFrame with features
        seq_length (int): Number of time steps in each sequence
        
    Returns:
        tuple: X (sequences) and y (targets)
    """
    X, y = [], []
    features = ['Close', 'RSI', 'volatility', 'momentum']
    
    # Get feature data and scale it
    feature_data = data[features].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    # Create target: 1 if next day's close is higher, -1 if lower
    target = (data['Close'].shift(-1) > data['Close']).astype(int) * 2 - 1
    target = target[:-1]  # Remove last row (no next day price)
    
    # Create sequences
    for i in range(len(scaled_data) - seq_length - 1):
        X.append(scaled_data[i:(i + seq_length)])
        y.append(target.iloc[i + seq_length])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(seq_length=60, n_features=4):
    """
    Build an LSTM model for binary classification
    
    Args:
        seq_length (int): Number of time steps in each sequence
        n_features (int): Number of features
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with dropout
    model.add(LSTM(units=50, return_sequences=True, 
                   input_shape=(seq_length, n_features)))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1, activation='tanh'))  # tanh for -1 to 1 range
    
    # Compile model with binary crossentropy loss and adam optimizer
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

def train_lstm_model(save_dir='models'):
    """
    Train LSTM model on stock data from 2019-2021
    
    Args:
        save_dir (str): Directory to save trained model
        
    Returns:
        tuple: Trained model and scaler
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
    
    # Create sequences for LSTM
    seq_length = 60
    X, y, scaler = create_sequences(combined_data, seq_length)
    print(f"Created {len(X)} sequences of length {seq_length}")
    
    # Split data into training and validation sets (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    # Build LSTM model
    model = build_lstm_model(seq_length, n_features=X.shape[2])
    print("LSTM model built")
    
    # Define callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'lstm_model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Train model
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and scaler
    model_path = os.path.join(save_dir, 'lstm_model.h5')
    scaler_path = os.path.join(save_dir, 'lstm_scaler.joblib')
    
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    print(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    # Save sequence length
    seq_length_path = os.path.join(save_dir, 'lstm_seq_length.joblib')
    joblib.dump(seq_length, seq_length_path)
    
    return model, scaler, history

if __name__ == "__main__":
    # Train the LSTM model
    model, scaler, history = train_lstm_model()
    print("LSTM model training completed successfully.")
