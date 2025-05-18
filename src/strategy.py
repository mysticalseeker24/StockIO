import pandas as pd
import numpy as np
import joblib
import os
from math import sqrt

def generate_signals(df, rf_model_path='../models/random_forest_model.joblib', scaler_path='../models/scaler.joblib'):
    """
    Generate trading signals using a hybrid approach that combines:
    1. Trend-Following (TF) signal based on SMA crossover
    2. Mean-Reversion (MR) signal based on Bollinger Bands
    3. Machine Learning (ML) signal from Random Forest model
    
    Args:
        df (DataFrame): DataFrame with processed stock data including technical indicators
        rf_model_path (str): Path to the trained Random Forest model
        scaler_path (str): Path to the fitted StandardScaler
        
    Returns:
        DataFrame: Original DataFrame with added signal columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df_signals = df.copy()
    
    # Generate Trend-Following signal (1 if uptrend, -1 if downtrend)
    # Based on 50-day SMA vs 200-day SMA (golden cross/death cross)
    df_signals['TF_signal'] = (df_signals['SMA50'] > df_signals['SMA200']).astype(int) * 2 - 1
    
    # Generate Mean-Reversion signal
    # 1 if price is below lower Bollinger Band (oversold, buy signal)
    # -1 if price is above upper Bollinger Band (overbought, sell signal)
    # 0 if price is between bands (no signal)
    df_signals['MR_signal'] = ((df_signals['Close'] < df_signals['BB_lower']).astype(int) - 
                              (df_signals['Close'] > df_signals['BB_upper']).astype(int))
    
    # Generate Machine Learning signal
    # Load the trained model and scaler
    try:
        model = joblib.load(rf_model_path)
        scaler = joblib.load(scaler_path)
        
        # Prepare features for prediction
        features = ['RSI', 'volatility', 'momentum', 'trend_strength']
        X = df_signals[features]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        df_signals['ML_signal'] = model.predict(X_scaled)
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading model or scaler: {e}")
        print("Using default ML signal of 0")
        df_signals['ML_signal'] = 0
    
    # Combine signals for final decision
    # Sum the three signals and determine direction:
    # Positive sum -> Buy (1), Negative sum -> Sell (-1)
    df_signals['final_signal'] = (df_signals['TF_signal'] + 
                                df_signals['MR_signal'] + 
                                df_signals['ML_signal']).apply(lambda x: 1 if x > 0 else -1)
    
    return df_signals

def simulate_trades(df, initial_capital=100000):
    """
    Simulate trades based on generated signals and calculate performance metrics.
    Implements a long-only strategy (buy when signal is 1, sell when signal is -1).
    
    Args:
        df (DataFrame): DataFrame with trading signals
        initial_capital (float): Starting capital for the simulation
        
    Returns:
        tuple: DataFrame with portfolio values and dict of performance metrics
    """
    # Make a copy to avoid modifying the original DataFrame
    df_trades = df.copy()
    
    # Initialize columns for portfolio tracking
    df_trades['position'] = 0  # 0: no position, 1: long position
    df_trades['shares'] = 0  # Number of shares held
    df_trades['cash'] = initial_capital  # Available cash
    df_trades['portfolio_value'] = initial_capital  # Total portfolio value
    df_trades['returns'] = 0.0  # Daily returns
    
    # Initialize trading variables
    position = 0  # No position initially
    cash = initial_capital  # Start with initial capital
    shares = 0  # No shares initially
    
    # Simulate trades for each day
    for i in range(1, len(df_trades)):
        prev_day = df_trades.iloc[i-1]
        current_day = df_trades.iloc[i]
        signal = current_day['final_signal']
        
        # Update position based on signal
        if signal == 1 and position == 0:  # Buy signal and not in position
            shares = cash // current_day['Close']  # Integer division for whole shares
            cash -= shares * current_day['Close']
            position = 1
        elif signal == -1 and position == 1:  # Sell signal and in position
            cash += shares * current_day['Close']
            shares = 0
            position = 0
        
        # Update portfolio value
        portfolio_value = cash + (shares * current_day['Close'])
        
        # Store values in DataFrame
        df_trades.iloc[i, df_trades.columns.get_loc('position')] = position
        df_trades.iloc[i, df_trades.columns.get_loc('shares')] = shares
        df_trades.iloc[i, df_trades.columns.get_loc('cash')] = cash
        df_trades.iloc[i, df_trades.columns.get_loc('portfolio_value')] = portfolio_value
        
        # Calculate daily return
        prev_portfolio = df_trades.iloc[i-1]['portfolio_value']
        df_trades.iloc[i, df_trades.columns.get_loc('returns')] = \
            (portfolio_value - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0
    
    # Calculate performance metrics
    final_value = df_trades.iloc[-1]['portfolio_value']
    total_return = (final_value - initial_capital) / initial_capital
    
    # Calculate annualized Sharpe Ratio (assuming 252 trading days per year)
    # Risk-free rate of 2% annually
    risk_free_daily = 0.02 / 252
    daily_returns = df_trades['returns'].dropna()
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    sharpe_ratio = (mean_daily_return - risk_free_daily) / std_daily_return * sqrt(252) \
        if std_daily_return > 0 else 0
    
    # Create dictionary of performance metrics
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': total_return * (252 / len(df_trades)),
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': (df_trades['portfolio_value'] / df_trades['portfolio_value'].cummax() - 1).min(),
        'win_rate': sum(df_trades['returns'] > 0) / len(df_trades['returns'].dropna())
    }
    
    return df_trades, metrics

def evaluate_strategy(stock_data, model_path='../models/random_forest_model.joblib', 
                     scaler_path='../models/scaler.joblib', initial_capital=100000):
    """
    Evaluate the trading strategy on a stock dataset.
    
    Args:
        stock_data (DataFrame): Processed stock data
        model_path (str): Path to the trained model
        scaler_path (str): Path to the fitted scaler
        initial_capital (float): Initial capital for simulation
        
    Returns:
        tuple: DataFrame with trades and dictionary of performance metrics
    """
    # Generate signals
    signals_df = generate_signals(stock_data, model_path, scaler_path)
    
    # Simulate trades
    trades_df, metrics = simulate_trades(signals_df, initial_capital)
    
    return trades_df, metrics

if __name__ == "__main__":
    # Example usage (requires data_loader module)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_and_process_data
    
    # Load data for a single stock
    data_dict = load_and_process_data()
    if data_dict:
        # Get the first stock's data
        stock_name = list(data_dict.keys())[0]
        stock_data = data_dict[stock_name]
        
        # Evaluate strategy
        trades_df, metrics = evaluate_strategy(stock_data)
        
        # Print results
        print(f"Evaluated strategy on {stock_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No data available. Please check the data directory.")
