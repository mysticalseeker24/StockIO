import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys

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
Test your trading strategy on historical stock data. Upload a CSV file with OHLCV data to see how our hybrid strategy performs.

**Features:**
- Trend following using SMA crossovers
- Mean reversion with Bollinger Bands
- Machine learning predictions with Random Forest
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
    
    # Convert Volume to numeric if it's a string with commas
    if isinstance(df['Volume'].iloc[0], str):
        df['Volume'] = df['Volume'].str.replace(',', '').astype(float).astype(int)
    
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
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            st.sidebar.success("File uploaded successfully!")
            
            # Display raw data preview
            with st.expander("Raw Data Preview"):
                st.dataframe(df.head())
            
            # Process the data
            processed_df = process_uploaded_data(df)
            
            if processed_df is not None:
                st.sidebar.info(f"Processed data: {len(processed_df)} rows")
                
                # Load model and scaler
                models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
                model_path = os.path.join(models_dir, 'random_forest_model.joblib')
                scaler_path = os.path.join(models_dir, 'scaler.joblib')
                
                # Check if model files exist
                model_exists = os.path.exists(model_path)
                scaler_exists = os.path.exists(scaler_path)
                
                if not model_exists or not scaler_exists:
                    st.warning("Model or scaler not found. Machine learning signals will not be used.")
                
                # Generate signals
                signals_df = generate_signals(processed_df, model_path, scaler_path)
                
                # Simulation parameters
                initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
                risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100
                stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5) / 100
                
                # Simulate trades with selected parameters
                trades_df, metrics = simulate_trades(signals_df, initial_capital, risk_per_trade, stop_loss_pct)
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Portfolio Value", "Trading Signals"])
                
                with tab1:
                    # Display metrics
                    st.header("Performance Metrics")
                    
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
                
                with tab2:
                    # Plot portfolio value
                    st.header("Portfolio Value Over Time")
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
                
                with tab3:
                    # Display signals table
                    st.header("Trading Signals and Positions")
                    signals_table = trades_df[['Open', 'High', 'Low', 'Close', 'TF_signal', 
                                              'MR_signal', 'ML_signal', 'final_signal', 
                                              'position', 'portfolio_value', 'stop_loss']].copy()
                    
                    # Add color coding based on signal
                    st.dataframe(signals_table.style.applymap(
                        lambda x: 'background-color: #90EE90' if x == 1 else 
                                ('background-color: #FFA07A' if x == -1 else ''), 
                        subset=['final_signal']
                    ))
                    
                    # Download link for the full results
                    st.download_button(
                        label="Download Full Results CSV",
                        data=trades_df.to_csv(),
                        file_name="stockio_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show example when no file is uploaded
        st.info("Please upload a CSV file with stock data to begin analysis.")
        st.markdown("""
        ### CSV File Format
        Your CSV file should have these columns:
        - `Date`: Date in YYYY-MM-DD format
        - `Open`: Opening price
        - `High`: Highest price of the day
        - `Low`: Lowest price of the day
        - `Close`: Closing price
        - `Volume`: Trading volume
        
        You can also test with example data from popular financial data providers.
        """)

if __name__ == "__main__":
    main()
