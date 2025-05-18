# StockIO

A trading strategy platform that uses historical price data to maximize alpha.

## Objective
Develop a trading strategy using historical price data to maximize alpha. The strategy is evaluated on performance (measured by Sharpe Ratio and other metrics) and the clarity of methodology.

## Features

- Technical Analysis: Implements indicators like Moving Averages, RSI, MACD, and custom indicators
- Statistical Models: Supports mean reversion, momentum strategies, and pairs trading
- Machine Learning: Includes predictive models for price movements and clustering for regime detection
- Entry/Exit Points: Calculates optimal entry and exit points for instruments
- Performance Metrics: Optimizes for Sharpe Ratio and other risk-adjusted return metrics

## Project Structure

- `data/`: Contains historical OHLCV data for selected instruments
- `src/`: Contains the source code for the project
- `models/`: Stores trained models

## Getting Started

```bash
# Clone the repository
git clone https://github.com/mysticalseeker24/StockIO.git

# Install dependencies
pip install -r requirements.txt
```

## License
MIT
