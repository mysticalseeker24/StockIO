import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys
import zipfile
import io
import tempfile
from datetime import datetime

# Add the parent directory to the path so we can import the local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_and_process_data
from src.strategy import generate_signals, simulate_trades

# Set page config
st.set_page_config(page_title="StockIO - Trading Strategy Tester", 
                   page_icon="📈", 
                   layout="wide")

# Title and description
st.title("StockIO - Alpha Strategy Builder")
st.markdown("""
Test your trading strategy on historical stock data. Upload CSV files or ZIP archives with OHLCV data to see how our hybrid strategy performs.

**Features:**
- Compare multiple stocks or strategies side by side
- Optimize parameters to maximize Sharpe ratio
- Target Sharpe ratios over 1.5, aiming for 3.0 (excellent performance)
- Risk management with position sizing and stop-loss
""")

# Helper function to process uploaded data
def process_uploaded_data(df):
    """
    Process the uploaded dataframe to generate features
    similar to data_loader.py
    """
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None
    
    # Handle Volume column which might contain commas, NaN values or strings
    try:
        # First check if Volume is already numeric
        if not pd.api.types.is_numeric_dtype(df['Volume']):
            df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
        # Fill NA values with 0 before converting to int
        df['Volume'] = df['Volume'].fillna(0).astype(int)
    except Exception as e:
        st.warning(f"Warning: Error processing Volume column: {e}")
        # Fallback: set Volume to 0 if conversion fails
        df['Volume'] = 0
    
    # Calculate returns
    df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Calculate SMA indicators using pandas rolling mean
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['trend_strength'] = (df['SMA50'] - df['SMA200']) / df['SMA200']
    
    # Calculate RSI using pandas (14-day period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate momentum
    df['momentum'] = df['returns'].rolling(window=10).sum()
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (std * 2)
    df['BB_lower'] = df['BB_middle'] - (std * 2)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# Main app function
def main():
    # Sidebar
    st.sidebar.header("Upload Data")
    upload_option = st.sidebar.radio("Upload option", ["Single CSV", "Multiple CSVs", "ZIP File"])
    
    # Results storage
    all_results = {}
    uploaded_files = []
    
    if upload_option == "Single CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            uploaded_files = [uploaded_file]
    
    elif upload_option == "Multiple CSVs":
        uploaded_files = st.sidebar.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    
    elif upload_option == "ZIP File":
        uploaded_zip = st.sidebar.file_uploader("Upload ZIP file", type="zip")
        if uploaded_zip:
            # Extract zip file contents
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                # Create a temporary directory to extract files
                with tempfile.TemporaryDirectory() as tmpdirname:
                    zip_ref.extractall(tmpdirname)
                    # Find all CSV files in the extracted directory
                    csv_files = []
                    for root, dirs, files in os.walk(tmpdirname):
                        for file in files:
                            if file.endswith(".csv"):
                                file_path = os.path.join(root, file)
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                    # Create a file-like object
                                    file_obj = io.BytesIO(file_content)
                                    file_obj.name = file  # Add name attribute for compatibility
                                    csv_files.append(file_obj)
                    uploaded_files = csv_files
                    st.sidebar.success(f"Found {len(csv_files)} CSV files in the ZIP archive.")
    
    if uploaded_files:
        # Display optimization parameters in the sidebar
        st.sidebar.header("Strategy Optimization")
        initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
        risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100
        stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5) / 100
        
        # Add parameter optimization option
        optimize_params = st.sidebar.checkbox("Optimize Strategy Parameters", value=True)
        
        # Load model and scaler
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        model_path = os.path.join(models_dir, 'random_forest_model.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        
        # Check if model files exist
        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)
        
        if not model_exists or not scaler_exists:
            st.warning("Model or scaler not found. Machine learning signals will not be used.")
        
        # Process each uploaded file
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Use filename as identifier
                file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else f"File_{i+1}"
                
                # Create a progress section for this file
                with st.spinner(f"Processing {file_name}"):
                    # Load the data
                    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
                    st.success(f"Loaded {file_name} successfully!")
                    
                    # Display raw data preview
                    with st.expander(f"Raw Data Preview - {file_name}"):
                        st.dataframe(df.head())
                    
                    # Process the data
                    processed_df = process_uploaded_data(df)
                    
                    if processed_df is not None:
                        st.info(f"Processed {file_name}: {len(processed_df)} rows")
                        
                        # Generate signals with optional parameter optimization
                        if optimize_params:
                            # Use parameter grid search to find optimal parameters
                            from src.strategy import optimize_strategy_parameters
                            best_params, signals_df, trades_df, metrics = optimize_strategy_parameters(
                                processed_df, model_path, scaler_path, initial_capital
                            )
                            st.success(f"Optimized parameters for {file_name}: {best_params}")
                        else:
                            # Use default parameters
                            signals_df = generate_signals(processed_df, model_path, scaler_path)
                            trades_df, metrics = simulate_trades(signals_df, initial_capital, risk_per_trade, stop_loss_pct)
                        
                        # Store results for comparison
                        all_results[file_name] = {
                            'trades_df': trades_df,
                            'metrics': metrics,
                            'processed_df': processed_df
                        }
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
        
        # Create tabs for different views - only if we have results
        if all_results:
            tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "Performance Metrics", "Portfolio Value", "Trading Signals"])
            
            # Sort results by Sharpe ratio
            sorted_results = sorted(all_results.items(), 
                                   key=lambda x: x[1]['metrics']['sharpe_ratio'],
                                   reverse=True)
            
            # Tab 1: Comparison of all stocks
            with tab1:
                st.header("Strategy Comparison")                
                # Create comparison table
                comparison_data = {
                    'Stock': [],
                    'Total Return': [],
                    'Sharpe Ratio': [],
                    'Max Drawdown': [],
                    'Win Rate': []
                }
                
                for name, result in sorted_results:
                    comparison_data['Stock'].append(name)
                    comparison_data['Total Return'].append(f"{result['metrics']['total_return']:.2%}")
                    comparison_data['Sharpe Ratio'].append(f"{result['metrics']['sharpe_ratio']:.2f}")
                    comparison_data['Max Drawdown'].append(f"{result['metrics']['max_drawdown']:.2%}")
                    comparison_data['Win Rate'].append(f"{result['metrics']['win_rate']:.2%}")
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Plot Sharpe ratios comparison
                st.subheader("Sharpe Ratio Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Convert Sharpe ratios from string to float
                sharpe_values = []
                for sr in comparison_data['Sharpe Ratio']:
                    try:
                        sharpe_values.append(float(sr))
                    except ValueError:
                        sharpe_values.append(0)
                
                bars = ax.bar(comparison_data['Stock'], sharpe_values)
                
                # Color bars based on Sharpe ratio values
                for i, bar in enumerate(bars):
                    sr_value = sharpe_values[i]
                    if sr_value >= 3.0:
                        bar.set_color('darkgreen')
                    elif sr_value >= 1.5:
                        bar.set_color('green')
                    elif sr_value >= 1.0:
                        bar.set_color('lightgreen')
                    elif sr_value >= 0:
                        bar.set_color('yellow')
                    else:
                        bar.set_color('red')
                
                ax.set_title('Sharpe Ratio Comparison')
                ax.set_xlabel('Stock')
                ax.set_ylabel('Sharpe Ratio')
                ax.axhline(y=1.5, color='r', linestyle='--', label='Target (1.5)')
                ax.axhline(y=3.0, color='g', linestyle='--', label='Excellent (3.0)')
                ax.grid(True)
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Select stock to view in detail
                selected_stock = st.selectbox(
                    "Select stock for detailed view", 
                    list(all_results.keys()),
                    index=0
                )
            
            # Get the selected stock data for other tabs
            if selected_stock in all_results:
                trades_df = all_results[selected_stock]['trades_df']
                metrics = all_results[selected_stock]['metrics']
                
                # Tab 2: Performance metrics                
                with tab2:
                    # Display metrics
                    st.header(f"Performance Metrics - {selected_stock}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    with col4:
                        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    
                    # Additional metrics table
                    st.subheader("Detailed Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    })
                    st.dataframe(metrics_df)
                
                with tab3:
                    # Plot portfolio value
                    st.header(f"Portfolio Value Over Time - {selected_stock}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(trades_df.index, trades_df['portfolio_value'])
                    ax.set_title('Portfolio Value')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Value ($)')
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Daily returns histogram
                    st.subheader("Distribution of Daily Returns")
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    trades_df['returns'].hist(bins=50, ax=ax2)
                    ax2.set_title('Daily Returns Distribution')
                    ax2.set_xlabel('Return')
                    ax2.set_ylabel('Frequency')
                    st.pyplot(fig2)
                
                with tab4:
                    # Display signals table
                    st.header(f"Trading Signals and Positions - {selected_stock}")
                    signals_table = trades_df[['Open', 'High', 'Low', 'Close', 'TF_signal', 
                                              'MR_signal', 'ML_signal', 'final_signal', 
                                              'position', 'portfolio_value', 'stop_loss']].copy()
                    
                    # Add color coding based on signal
                    st.dataframe(signals_table.style.map(
                        lambda x: 'background-color: #90EE90' if x == 1 else 
                                ('background-color: #FFA07A' if x == -1 else ''), 
                        subset=['final_signal']
                    ))
                    
                    # Download link for the full results
                    st.download_button(
                        label="Download Full Results CSV",
                        data=trades_df.to_csv(),
                        file_name=f"stockio_results_{selected_stock}.csv",
                        mime="text/csv"
                    )
        
        else:
            # Show example when no file is uploaded
            st.info("Please upload a CSV file with stock data to begin analysis.")
            st.markdown("""
            ### CSV File Format
            Your CSV file should have these columns:
            - `Date`: Date in YYYY-MM-DD format
            - `Open`: Opening price
            - `High`: High price
            - `Low`: Low price
            - `Close`: Closing price
            - `Volume`: Trading volume
            """)
    else:
        # Show example when no file is uploaded
        st.info("Please upload a CSV file with stock data to begin analysis.")
        st.markdown("""
        ### CSV File Format
        Your CSV file should have these columns:
        - `Date`: Date in YYYY-MM-DD format
        - `Open`: Opening price
        - `High`: High price
        - `Low`: Low price
        - `Close`: Closing price
        - `Volume`: Trading volume
        """)

if __name__ == "__main__":
    main()
