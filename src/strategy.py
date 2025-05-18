import pandas as pd
import numpy as np
import joblib
import os
from math import sqrt
from itertools import product

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

def simulate_trades(df, initial_capital=100000, risk_per_trade=0.01, stop_loss_pct=0.02):
    """
    Simulate trades based on generated signals and calculate performance metrics.
    Implements a long-only strategy with volatility-based position sizing and stop-loss.
    
    Args:
        df (DataFrame): DataFrame with trading signals
        initial_capital (float): Starting capital for the simulation
        risk_per_trade (float): Percentage of capital to risk per trade (0.01 = 1%)
        stop_loss_pct (float): Stop loss percentage (0.02 = 2%)
        
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
    df_trades['stop_loss'] = 0.0  # Stop loss price level
    
    # Initialize trading variables
    position = 0  # No position initially
    cash = initial_capital  # Start with initial capital
    shares = 0  # No shares initially
    entry_price = 0  # Price at which position was entered
    stop_loss_price = 0  # Stop loss price level
    
    # Simulate trades for each day
    for i in range(1, len(df_trades)):
        prev_day = df_trades.iloc[i-1]
        current_day = df_trades.iloc[i]
        signal = current_day['final_signal']
        current_price = current_day['Close']
        
        # Check if stop loss is triggered
        stop_loss_triggered = False
        if position == 1 and current_price <= stop_loss_price:
            # Stop loss triggered - sell position
            cash += shares * current_price
            shares = 0
            position = 0
            stop_loss_triggered = True
            print(f"Stop loss triggered at {current_price:.2f}")
        
        # Update position based on signal if stop loss wasn't triggered
        if not stop_loss_triggered:
            if signal == 1 and position == 0:  # Buy signal and not in position
                # Volatility-based position sizing
                # Use ATR (Average True Range) or volatility for risk calculation
                # If not available, use a simplified approach with recent volatility
                price_volatility = current_day['volatility'] * current_price
                if price_volatility > 0:
                    # Calculate position size based on risk
                    risk_amount = initial_capital * risk_per_trade  # Amount willing to risk
                    position_size = risk_amount / (price_volatility * 2)  # Simplified risk calculation
                    shares = int(position_size)  # Integer number of shares
                else:
                    # Fallback if volatility is zero or not available
                    shares = int(cash * 0.95 // current_price)  # Use 95% of available cash
                
                if shares > 0:
                    entry_price = current_price
                    cash -= shares * entry_price
                    position = 1
                    
                    # Set stop loss price
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
            
            elif signal == -1 and position == 1:  # Sell signal and in position
                cash += shares * current_price
                shares = 0
                position = 0
                stop_loss_price = 0
        
        # Update portfolio value
        portfolio_value = cash + (shares * current_price)
        
        # Store values in DataFrame (convert to appropriate types first)
        df_trades.iloc[i, df_trades.columns.get_loc('position')] = position
        df_trades.iloc[i, df_trades.columns.get_loc('shares')] = shares
        df_trades.iloc[i, df_trades.columns.get_loc('cash')] = float(cash)
        df_trades.iloc[i, df_trades.columns.get_loc('portfolio_value')] = float(portfolio_value)
        df_trades.iloc[i, df_trades.columns.get_loc('stop_loss')] = float(stop_loss_price)
        
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

def optimize_parameters(df, param_grid=None, train_period=None):
    """
    Optimize strategy parameters using grid search to maximize Sharpe Ratio.
    
    Args:
        df (DataFrame): Stock data with OHLCV data
        param_grid (dict): Dictionary of parameter grids to search
        train_period (tuple): Start and end dates for training period (str: 'YYYY-MM-DD')
        
    Returns:
        dict: Optimal parameters and performance metrics
    """
    if param_grid is None:
        # Default parameter grid to search
        param_grid = {
            'sma_fast': [20, 50, 100],
            'sma_slow': [100, 200, 300],
            'rsi_period': [10, 14, 20],
            'bb_period': [15, 20, 25],
            'bb_stdev': [1.5, 2.0, 2.5],
            'risk_per_trade': [0.005, 0.01, 0.02],
            'stop_loss_pct': [0.01, 0.02, 0.03]
        }
    
    # Filter data to training period if specified
    if train_period is not None:
        train_start, train_end = train_period
        train_df = df.loc[train_start:train_end].copy()
    else:
        train_df = df.copy()
    
    if train_df.empty:
        raise ValueError("Training data is empty. Check your date range.")
    
    best_sharpe = -float('inf')
    best_params = {}
    best_metrics = {}
    
    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['sma_fast'],
        param_grid['sma_slow'],
        param_grid['rsi_period'],
        param_grid['bb_period'],
        param_grid['bb_stdev'],
        param_grid['risk_per_trade'],
        param_grid['stop_loss_pct']
    ))
    
    print(f"Optimizing parameters with {len(param_combinations)} combinations...")
    
    # Test each parameter combination
    for i, (sma_fast, sma_slow, rsi_period, bb_period, bb_stdev, risk_per_trade, stop_loss_pct) in enumerate(param_combinations):
        # Skip invalid combinations
        if sma_fast >= sma_slow:
            continue
        
        # Generate features with current parameters
        test_df = train_df.copy()
        
        # Calculate SMA indicators
        test_df['SMA50'] = test_df['Close'].rolling(window=sma_fast).mean()
        test_df['SMA200'] = test_df['Close'].rolling(window=sma_slow).mean()
        
        # Calculate trend strength
        test_df['trend_strength'] = (test_df['SMA50'] - test_df['SMA200']) / test_df['SMA200']
        
        # Calculate RSI
        delta = test_df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        test_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        test_df['BB_middle'] = test_df['Close'].rolling(window=bb_period).mean()
        std = test_df['Close'].rolling(window=bb_period).std()
        test_df['BB_upper'] = test_df['BB_middle'] + (std * bb_stdev)
        test_df['BB_lower'] = test_df['BB_middle'] - (std * bb_stdev)
        
        # Drop NaN values
        test_df.dropna(inplace=True)
        
        if test_df.empty:
            continue
        
        # Generate signals
        signals_df = generate_signals(test_df)
        
        # Simulate trades with current parameters
        try:
            _, metrics = simulate_trades(signals_df, initial_capital=100000,
                                       risk_per_trade=risk_per_trade,
                                       stop_loss_pct=stop_loss_pct)
            
            # Check if this parameter set gives better Sharpe ratio
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = {
                    'sma_fast': sma_fast,
                    'sma_slow': sma_slow,
                    'rsi_period': rsi_period,
                    'bb_period': bb_period,
                    'bb_stdev': bb_stdev,
                    'risk_per_trade': risk_per_trade,
                    'stop_loss_pct': stop_loss_pct
                }
                best_metrics = metrics
                
                print(f"New best parameters found [{i+1}/{len(param_combinations)}]:")
                print(f"  Sharpe Ratio: {best_sharpe:.4f}")
                print(f"  Parameters: {best_params}")
        except Exception as e:
            print(f"Error with parameters {i+1}/{len(param_combinations)}: {e}")
    
    return {
        'best_params': best_params,
        'best_metrics': best_metrics
    }

def evaluate_strategy(stock_data, model_path='../models/random_forest_model.joblib', 
                     scaler_path='../models/scaler.joblib', initial_capital=100000,
                     risk_per_trade=0.01, stop_loss_pct=0.02):
    """
    Evaluate the trading strategy on a stock dataset.
    
    Args:
        stock_data (DataFrame): Processed stock data
        model_path (str): Path to the trained model
        scaler_path (str): Path to the fitted scaler
        initial_capital (float): Initial capital for simulation
        risk_per_trade (float): Percentage of capital to risk per trade
        stop_loss_pct (float): Stop loss percentage
        
    Returns:
        tuple: DataFrame with trades and dictionary of performance metrics
    """
    # Generate signals
    signals_df = generate_signals(stock_data, model_path, scaler_path)
    
    # Simulate trades
    trades_df, metrics = simulate_trades(signals_df, initial_capital, risk_per_trade, stop_loss_pct)
    
    return trades_df, metrics

def optimize_strategy_parameters(df, model_path='../models/random_forest_model.joblib', 
                           scaler_path='../models/scaler.joblib', initial_capital=100000):
    """
    Optimize strategy parameters to maximize Sharpe Ratio, targeting values above 1.5.
    
    Args:
        df (DataFrame): Processed stock data with technical indicators
        model_path (str): Path to the trained model
        scaler_path (str): Path to the fitted scaler
        initial_capital (float): Initial capital for trading simulation
        
    Returns:
        tuple: Best parameters, signals df, trades df, and performance metrics
    """
    # Generate base signals
    signals_df = generate_signals(df, model_path, scaler_path)
    
    # Define parameter grid for optimization
    param_grid = {
        'risk_per_trade': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],  # 0.5% to 3% risk per trade
        'stop_loss_pct': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]  # 1% to 5% stop loss
    }
    
    # Track best parameters and Sharpe ratio
    best_sharpe = -float('inf')
    best_params = {}
    best_trades_df = None
    best_metrics = {}
    
    # Grid search through parameter combinations
    combinations = list(product(param_grid['risk_per_trade'], param_grid['stop_loss_pct']))
    total_combinations = len(combinations)
    
    # Try each parameter combination
    for i, (risk_per_trade, stop_loss_pct) in enumerate(combinations):
        # Simulate trades with current parameters
        trades_df, metrics = simulate_trades(signals_df, initial_capital, risk_per_trade, stop_loss_pct)
        
        # Check if this is the best Sharpe ratio so far
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = {'risk_per_trade': risk_per_trade, 'stop_loss_pct': stop_loss_pct}
            best_trades_df = trades_df.copy()
            best_metrics = metrics.copy()
    
    # If we haven't found a good Sharpe ratio (>1.5), try more aggressive parameters
    if best_sharpe < 1.5:
        # Try more aggressive parameter combinations focusing on higher conviction trades
        aggressive_param_grid = {
            'risk_per_trade': [0.03, 0.04, 0.05],  # Higher risk per trade
            'stop_loss_pct': [0.07, 0.08, 0.1]    # Wider stop losses
        }
        
        aggressive_combinations = list(product(aggressive_param_grid['risk_per_trade'], aggressive_param_grid['stop_loss_pct']))
        
        # Try each aggressive parameter combination
        for risk_per_trade, stop_loss_pct in aggressive_combinations:
            # Simulate trades with current parameters
            trades_df, metrics = simulate_trades(signals_df, initial_capital, risk_per_trade, stop_loss_pct)
            
            # Check if this is the best Sharpe ratio so far
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = {'risk_per_trade': risk_per_trade, 'stop_loss_pct': stop_loss_pct}
                best_trades_df = trades_df.copy()
                best_metrics = metrics.copy()
    
    return best_params, signals_df, best_trades_df, best_metrics

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
        
        # Optional: Optimize parameters first
        # optimize_result = optimize_parameters(stock_data, train_period=('2019-01-01', '2019-12-31'))
        # print(f"Optimal parameters: {optimize_result['best_params']}")
        
        # Evaluate strategy
        trades_df, metrics = evaluate_strategy(stock_data)
        
        # Print results
        print(f"Evaluated strategy on {stock_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No data available. Please check the data directory.")
