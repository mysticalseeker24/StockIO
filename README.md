# StockIO

A trading strategy platform that uses historical price data to maximize alpha.

## Objective

Develop a trading strategy using historical price data to maximize alpha. The strategy is evaluated on performance (measured by Sharpe Ratio and other metrics) and the clarity of methodology.

## Features

- **LSTM Deep Learning**: Incorporates a Long Short-Term Memory neural network for advanced price movement forecasting
- **Adaptive Position Sizing**: Dynamically adjusts position sizes based on signal conviction and agreement
- **Technical Analysis**: Implements indicators like Moving Averages, RSI, and Bollinger Bands using pandas/numpy (no Ta-Lib dependency)
- **Hybrid Strategy**: Combines trend-following, mean-reversion, and machine learning signals for better performance
- **Enhanced Risk Management**: Includes volatility-based position sizing with adaptive scaling and optimized stop-loss implementation
- **Parameter Optimization**: Sophisticated grid search targeting Sharpe Ratios above 2.0, with extended parameter ranges
- **Multi-File Comparison**: Supports analysis of multiple stocks simultaneously via individual files or ZIP archives
- **Interactive UI**: Streamlit-based frontend with comparative analysis dashboards and enhanced visualization
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
2. **Upload Data**: Choose from three upload options:
   - Single CSV file
   - Multiple CSV files at once
   - ZIP archive containing multiple CSV files
3. **Adjust Risk Parameters**:
   - Risk per trade: Controls position sizing (default: 1%)
   - Stop-loss percentage: Sets maximum loss per trade (default: 2%)
   - Enable automatic parameter optimization to maximize Sharpe ratio
4. **Compare Results**: View side-by-side comparison of multiple stocks
5. **Analyze Details**: Select individual stocks for in-depth performance analysis

## Implementation Details

### Data Processing

The `data_loader.py` module implements the following technical indicators without Ta-Lib:

- **Simple Moving Averages (SMA50, SMA200)**: Using pandas rolling functions
- **RSI (Relative Strength Index)**: Custom implementation with average gains/losses
- **Bollinger Bands**: Using rolling mean and standard deviation
- **Additional Features**: Trend strength, volatility, and momentum indicators

### Advanced Machine Learning Models

The system uses two complementary machine learning approaches:

1. **Random Forest Model** (`model_trainer.py`):
   - Trained on technical indicators to classify price movements
   - Uses sklearn's RandomForestClassifier with optimized parameters
   - Feature importance analysis to identify key predictors

2. **LSTM Neural Network** (`lstm_trainer.py`):
   - 60-day sequence input for temporal pattern recognition
   - Two LSTM layers (50 units each) for deep pattern recognition
   - Advanced time-series forecasting of price movements
   - Trained on multiple stocks for robust generalization

### Enhanced Hybrid Strategy

The hybrid strategy now combines four signal generators:

1. **Trend-Following**: Based on SMA50 vs SMA200 crossovers
2. **Mean-Reversion**: Uses Bollinger Bands for oversold/overbought conditions
3. **Random Forest**: Machine learning classification based on technical indicators
4. **LSTM Deep Learning**: Sequence-based neural network predictions

Advanced risk management includes:

- Adaptive position sizing based on signal conviction
- Dynamic volatility-scaled risk adjustment
- Optimized stop-loss parameters targeting Sharpe Ratios > 2.0
- Signal agreement analysis to boost conviction on high-probability trades

## License

MIT
