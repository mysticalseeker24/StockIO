import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_packages = []
    
    # Essential packages
    essential_packages = ["pandas", "numpy", "matplotlib"]
    for package in essential_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Optional packages
    optional_packages = [
        ("talib", "Technical analysis library for indicators"),
        ("sklearn", "Machine learning functionality"),
        ("streamlit", "Web interface"),
        ("joblib", "Model saving/loading")
    ]
    
    optional_missing = []
    for package, description in optional_packages:
        try:
            __import__(package)
        except ImportError:
            optional_missing.append((package, description))
    
    return missing_packages, optional_missing

def process_data_without_talib(df):
    """Process data without using talib (fallback function)"""
    # Calculate returns
    df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Simple Moving Averages (using pandas)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Trend strength
    df['trend_strength'] = (df['SMA50'] - df['SMA200']) / df['SMA200']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Momentum
    df['momentum'] = df['returns'].rolling(window=10).sum()
    
    # Simple RSI implementation without talib
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple Bollinger Bands implementation
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std * 2)
    df['BB_lower'] = df['BB_middle'] - (std * 2)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def load_data():
    """Load stock data from the data directory"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    data_dict = {}
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return data_dict
    
    # List CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return data_dict
    
    print(f"Found {len(csv_files)} CSV files in the data directory")
    
    # Try to import talib, use fallback if not available
    use_talib = False
    try:
        import talib
        use_talib = True
        print("Using TA-Lib for technical indicators")
    except ImportError:
        print("TA-Lib not found, using pandas-based calculations for indicators")
    
    # Process each CSV file
    for filename in csv_files:
        try:
            # Extract stock name from filename
            stock_name = filename.replace('.csv', '')
            
            # Load CSV file
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            
            # Select required columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Convert Volume to numeric if it's a string
            if isinstance(df['Volume'].iloc[0], str):
                df['Volume'] = df['Volume'].str.replace(',', '').astype(float).astype(int)
            
            # Process the data
            if use_talib:
                # Import again within the try block
                import talib
                from src.data_loader import load_and_process_data
                
                # Calculate returns
                df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
                
                # Calculate SMA indicators
                df['SMA50'] = talib.SMA(df['Close'].values, timeperiod=50)
                df['SMA200'] = talib.SMA(df['Close'].values, timeperiod=200)
                
                # Calculate trend strength
                df['trend_strength'] = (df['SMA50'] - df['SMA200']) / df['SMA200']
                
                # Calculate RSI
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
                
                # Calculate volatility
                df['volatility'] = df['returns'].rolling(window=20).std()
                
                # Calculate momentum
                df['momentum'] = df['returns'].rolling(window=10).sum()
                
                # Calculate Bollinger Bands
                df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
                    df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            else:
                # Use the fallback function
                df = process_data_without_talib(df)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Add stock name column
            df['stock_name'] = stock_name
            
            # Store in dictionary
            data_dict[stock_name] = df
            print(f"Processed {stock_name}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return data_dict

def train_model(data_dict):
    """Train a Random Forest model if sklearn is available"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Combine data for training period (2019-2021)
        combined_dfs = []
        for stock_name, df in data_dict.items():
            try:
                # Filter by date range
                filtered_df = df.loc['2019-01-01':'2021-12-31'].copy()
                if not filtered_df.empty:
                    combined_dfs.append(filtered_df)
            except Exception as e:
                print(f"Error filtering {stock_name}: {str(e)}")
        
        if not combined_dfs:
            print("No data available for the specified date range (2019-2021)")
            return None, None
        
        # Concatenate all DataFrames
        combined_data = pd.concat(combined_dfs)
        combined_data.sort_index(inplace=True)
        
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
        model_path = os.path.join(models_dir, 'random_forest_model.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        
        print(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        print(f"Saving scaler to {scaler_path}")
        joblib.dump(scaler, scaler_path)
        
        return model, scaler
    
    except ImportError:
        print("scikit-learn not available. Skipping model training.")
        return None, None

def generate_signals(df, model=None, scaler=None):
    """Generate trading signals using a hybrid approach"""
    # Make a copy to avoid modifying the original DataFrame
    df_signals = df.copy()
    
    # Generate Trend-Following signal (1 if uptrend, -1 if downtrend)
    df_signals['TF_signal'] = (df_signals['SMA50'] > df_signals['SMA200']).astype(int) * 2 - 1
    
    # Generate Mean-Reversion signal
    df_signals['MR_signal'] = ((df_signals['Close'] < df_signals['BB_lower']).astype(int) - 
                             (df_signals['Close'] > df_signals['BB_upper']).astype(int))
    
    # Generate Machine Learning signal if model is available
    if model is not None and scaler is not None:
        try:
            # Prepare features for prediction
            features = ['RSI', 'volatility', 'momentum', 'trend_strength']
            X = df_signals[features]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            df_signals['ML_signal'] = model.predict(X_scaled)
        except Exception as e:
            print(f"Error generating ML signals: {str(e)}")
            df_signals['ML_signal'] = 0
    else:
        df_signals['ML_signal'] = 0
        
    # Combine signals
    df_signals['final_signal'] = (df_signals['TF_signal'] + 
                               df_signals['MR_signal'] + 
                               df_signals['ML_signal']).apply(lambda x: 1 if x > 0 else -1)
    
    return df_signals

def simulate_trades(df, initial_capital=100000):
    """Simulate trades based on generated signals"""
    from src.strategy import simulate_trades
    try:
        return simulate_trades(df, initial_capital)
    except Exception as e:
        print(f"Error in trade simulation: {str(e)}")
        # Fallback implementation
        # Similar to the one in strategy.py but with minimal dependencies
        df_trades = df.copy()
        df_trades['position'] = 0
        df_trades['shares'] = 0
        df_trades['cash'] = initial_capital
        df_trades['portfolio_value'] = initial_capital
        df_trades['returns'] = 0.0
        
        position = 0
        cash = initial_capital
        shares = 0
        
        for i in range(1, len(df_trades)):
            prev_day = df_trades.iloc[i-1]
            current_day = df_trades.iloc[i]
            signal = current_day['final_signal']
            
            if signal == 1 and position == 0:
                shares = cash // current_day['Close']
                cash -= shares * current_day['Close']
                position = 1
            elif signal == -1 and position == 1:
                cash += shares * current_day['Close']
                shares = 0
                position = 0
            
            portfolio_value = cash + (shares * current_day['Close'])
            
            df_trades.iloc[i, df_trades.columns.get_loc('position')] = position
            df_trades.iloc[i, df_trades.columns.get_loc('shares')] = shares
            df_trades.iloc[i, df_trades.columns.get_loc('cash')] = cash
            df_trades.iloc[i, df_trades.columns.get_loc('portfolio_value')] = portfolio_value
            
            prev_portfolio = df_trades.iloc[i-1]['portfolio_value']
            df_trades.iloc[i, df_trades.columns.get_loc('returns')] = \
                (portfolio_value - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0
        
        final_value = df_trades.iloc[-1]['portfolio_value']
        total_return = (final_value - initial_capital) / initial_capital
        
        from math import sqrt
        risk_free_daily = 0.02 / 252
        daily_returns = df_trades['returns'].dropna()
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        sharpe_ratio = (mean_daily_return - risk_free_daily) / std_daily_return * sqrt(252) \
            if std_daily_return > 0 else 0
        
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

def run_backtest(stock_name, df, model=None, scaler=None, initial_capital=100000):
    """Run backtesting for a single stock"""
    print(f"\nRunning backtest for {stock_name}")
    
    # Generate signals
    signals_df = generate_signals(df, model, scaler)
    
    # Simulate trades
    try:
        trades_df, metrics = simulate_trades(signals_df, initial_capital)
        
        # Print results
        print(f"Backtest results for {stock_name}:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        # Plot portfolio value
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df.index, trades_df['portfolio_value'])
        plt.title(f'Portfolio Value - {stock_name}')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'{stock_name}_portfolio.png'))
        print(f"Plot saved to results/{stock_name}_portfolio.png")
        
        # Save results to CSV
        trades_df.to_csv(os.path.join(results_dir, f'{stock_name}_trades.csv'))
        print(f"Trade data saved to results/{stock_name}_trades.csv")
        
        return trades_df, metrics
        
    except Exception as e:
        print(f"Error in backtest for {stock_name}: {str(e)}")
        return None, None

def run_streamlit():
    """Run the Streamlit app if available"""
    try:
        import streamlit
        print("Starting Streamlit app...")
        os.system("streamlit run src/app.py")
    except ImportError:
        print("Streamlit not installed. Skipping web interface launch.")
        print("To install Streamlit, run: pip install streamlit==1.22.0")

def main():
    """Main entry point"""
    print("\n===== StockIO - Alpha Strategy Builder =====\n")
    
    # Check dependencies
    missing, optional_missing = check_dependencies()
    
    if missing:
        print("Error: Missing essential dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nPlease install these packages using: pip install <package_name>")
        return
    
    if optional_missing:
        print("Warning: Some optional dependencies are missing:")
        for package, description in optional_missing:
            print(f"  - {package}: {description}")
        print("Some functionality may be limited.\n")
    
    # Load data
    print("Loading and processing stock data...")
    data_dict = load_data()
    
    if not data_dict:
        print("No data was loaded. Please make sure CSV files are in the data directory.")
        return
    
    # Train model if possible
    print("\nTraining model...")
    model, scaler = train_model(data_dict)
    
    # Run backtesting for each stock
    results = {}
    for stock_name, df in data_dict.items():
        trades_df, metrics = run_backtest(stock_name, df, model, scaler)
        if metrics:
            results[stock_name] = metrics
    
    # Display summary
    if results:
        print("\n===== Summary of Results =====\n")
        print(f"{'Stock':<20} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
        print("-" * 65)
        for stock_name, metrics in results.items():
            print(f"{stock_name:<20} {metrics['total_return']:.2%:<15} {metrics['sharpe_ratio']:.2f:<15} {metrics['max_drawdown']:.2%:<15}")
    
    # Run Streamlit app if available
    print("\nWould you like to start the web interface (Streamlit app)? (y/n)")
    choice = input("Enter your choice: ").strip().lower()
    if choice == 'y':
        run_streamlit()
    else:
        print("Skipping web interface launch.")
    
    print("\nStockIO execution completed.")

if __name__ == "__main__":
    main()
