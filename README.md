# StockIO

A trading strategy platform that uses historical price data to maximize alpha.

## Objective

Develop a trading strategy using historical price data to maximize alpha. The strategy is evaluated on performance (measured by Sharpe Ratio and other metrics) and the clarity of methodology.

## Features

- **Technical Analysis**: Implements indicators like Moving Averages, RSI, and Bollinger Bands using pandas/numpy (no Ta-Lib dependency)
- **Hybrid Strategy**: Combines trend-following, mean-reversion, and machine learning signals for better performance
- **Risk Management**: Includes volatility-based position sizing and stop-loss implementation
- **Parameter Optimization**: Uses grid search to maximize Sharpe Ratio
- **Interactive UI**: Streamlit-based frontend for testing strategies on uploaded datasets
- **Performance Metrics**: Calculates total return, annualized return, Sharpe ratio, max drawdown, and win rate

## Project Structure

- `data/`: Contains historical OHLCV data for selected instruments
- `src/`: Contains the source code for the project
  - `data_loader.py`: Handles loading and processing stock data with technical indicators
  - `model_trainer.py`: Trains a Random Forest model for price prediction
  - `strategy.py`: Implements the hybrid trading strategy with risk management
  - `app.py`: Streamlit frontend for visualizing strategy performance
- `models/`: Stores trained machine learning models and scalers

## Getting Started

```bash
# Clone the repository
git clone https://github.com/mysticalseeker24/StockIO.git

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/model_trainer.py

# Run the Streamlit app
streamlit run src/app.py
```

## Using the Application

1. **Start the Streamlit App**: Run `streamlit run src/app.py`
2. **Upload Data**: Use the file uploader in the sidebar to upload CSV files with OHLCV data
3. **Adjust Risk Parameters**:
   - Risk per trade: Controls position sizing (default: 1%)
   - Stop-loss percentage: Sets maximum loss per trade (default: 2%)
4. **View Results**: Explore performance metrics, portfolio value chart, and trading signals

## Implementation Details

### Data Processing

The `data_loader.py` module implements the following technical indicators without Ta-Lib:

- **Simple Moving Averages (SMA50, SMA200)**: Using pandas rolling functions
- **RSI (Relative Strength Index)**: Custom implementation with average gains/losses
- **Bollinger Bands**: Using rolling mean and standard deviation
- **Additional Features**: Trend strength, volatility, and momentum indicators

### Trading Strategy

The hybrid strategy combines three signal generators:

1. **Trend-Following**: Based on SMA50 vs SMA200 crossovers
2. **Mean-Reversion**: Uses Bollinger Bands for oversold/overbought conditions
3. **Machine Learning**: RandomForest model trained on technical indicators

Risk management includes:

- Volatility-based position sizing (% risk per trade)
- Stop-loss implementation (exit if loss exceeds threshold)
- Parameter optimization to maximize Sharpe Ratio

## License

MIT
