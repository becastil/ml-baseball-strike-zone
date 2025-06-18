# FinTech Terminal ML Engine

A comprehensive machine learning and AI module for financial analysis, prediction, and automated trading strategy development.

## Overview

The ML Engine provides state-of-the-art machine learning capabilities for financial markets, including:

- **Price Prediction**: LSTM-based neural networks for forecasting stock prices
- **Trend Classification**: Ensemble models for identifying market trends
- **Technical Analysis**: Comprehensive indicators and pattern recognition
- **Sentiment Analysis**: NLP-based analysis of news and social media
- **Trading Strategies**: Momentum-based strategies with backtesting
- **Risk Management**: Position sizing and portfolio optimization
- **Model Management**: Version control and deployment tools

## Features

### 1. Predictive Models
- **LSTM Price Predictor**: Deep learning model for time series forecasting
- **Trend Classifier**: Random Forest, XGBoost, and ensemble models
- **Volatility Forecasting**: GARCH and neural network models
- **Market Regime Detection**: Hidden Markov Models and clustering

### 2. Data Processing
- **Automated Data Collection**: Integration with Yahoo Finance, NewsAPI
- **Feature Engineering**: 100+ technical indicators and market microstructure features
- **Data Preprocessing**: Outlier detection, missing value imputation, normalization
- **Time Series Handling**: Sequence generation, rolling windows, resampling

### 3. Analysis Tools
- **Technical Analysis**: Moving averages, oscillators, volatility indicators
- **Pattern Recognition**: Candlestick patterns, support/resistance levels
- **Sentiment Analysis**: VADER, TextBlob, and FinBERT integration
- **Market Microstructure**: Spread analysis, volume profiling, order flow

### 4. Trading Strategies
- **Momentum Strategy**: Trend-following with dynamic position sizing
- **Mean Reversion**: Statistical arbitrage opportunities
- **Pairs Trading**: Cointegration-based strategies
- **ML-Based Strategies**: Using predictions for signal generation

### 5. Backtesting & Evaluation
- **Realistic Simulation**: Commission, slippage, and market impact modeling
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate
- **Risk Analytics**: VaR, CVaR, portfolio optimization
- **Visualization**: Comprehensive plotting and reporting

## Installation

1. Clone the repository:
```bash
cd fintech-terminal
```

2. Install dependencies:
```bash
cd ml_engine
pip install -r requirements.txt
```

3. Download NLTK data (for sentiment analysis):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## Quick Start

### 1. Price Prediction

```python
from src.predictors import PricePredictor
from src.processors import DataPreprocessor

# Initialize components
preprocessor = DataPreprocessor()
predictor = PricePredictor(sequence_length=60, lstm_units=[128, 64, 32])

# Fetch and prepare data
data = preprocessor.fetch_data('AAPL', start_date='2022-01-01')
ml_data = preprocessor.prepare_ml_data(data, target_column='close', sequence_length=60)

# Train model
predictor.train(
    ml_data['X_train'], 
    ml_data['y_train'],
    ml_data['X_val'], 
    ml_data['y_val'],
    epochs=100
)

# Make predictions
predictions = predictor.predict(ml_data['X_test'])
```

### 2. Technical Analysis

```python
from src.analysis import TechnicalAnalysis

# Initialize analyzer
analyzer = TechnicalAnalysis()

# Add all indicators
data_with_indicators = analyzer.analyze(data, indicators=['all'])

# Generate trading signals
signals = analyzer.generate_signals(data_with_indicators, strategy='combined')
```

### 3. Sentiment Analysis

```python
from src.analysis import SentimentAnalyzer

# Initialize analyzer
sentiment = SentimentAnalyzer(use_vader=True, use_finbert=True)

# Analyze news sentiment
news_sentiment = sentiment.analyze_news('AAPL', days_back=7)

# Get sentiment score
score = sentiment.calculate_sentiment_score('AAPL')
```

### 4. Backtesting

```python
from src.utils import Backtester
from src.strategies import MomentumStrategy

# Generate strategy signals
strategy = MomentumStrategy(lookback_period=20)
signals = strategy.generate_signals(data)

# Run backtest
backtester = Backtester(initial_capital=100000)
results = backtester.run(data, signals)

# View results
print(f"Total Return: {results.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
```

## Project Structure

```
ml_engine/
├── configs/
│   └── model_config.yaml      # Configuration files
├── data/                      # Local data cache
├── models/                    # Trained model storage
├── notebooks/
│   └── price_prediction_experiment.ipynb  # Example notebooks
├── src/
│   ├── __init__.py
│   ├── analysis/             # Technical and sentiment analysis
│   │   ├── technical_analysis.py
│   │   └── sentiment_analyzer.py
│   ├── predictors/           # ML prediction models
│   │   ├── price_predictor.py
│   │   └── trend_classifier.py
│   ├── processors/           # Data processing
│   │   ├── data_preprocessor.py
│   │   └── feature_engineer.py
│   ├── strategies/           # Trading strategies
│   │   └── momentum_strategy.py
│   └── utils/                # Utilities
│       ├── model_manager.py
│       └── backtester.py
├── requirements.txt
└── README.md
```

## Configuration

The ML Engine uses YAML configuration files for easy customization:

```yaml
# configs/model_config.yaml
models:
  price_predictor:
    type: lstm
    architecture:
      sequence_length: 60
      lstm_units: [128, 64, 32]
      dropout_rate: 0.2
    training:
      epochs: 100
      batch_size: 32
      learning_rate: 0.001
```

## API Reference

### Predictors

#### PricePredictor
```python
predictor = PricePredictor(
    sequence_length=60,    # Input sequence length
    n_features=10,         # Number of features
    lstm_units=[128, 64],  # LSTM layer sizes
    dropout_rate=0.2       # Dropout for regularization
)
```

#### TrendClassifier
```python
classifier = TrendClassifier(
    model_type='ensemble',   # 'rf', 'gb', 'xgb', or 'ensemble'
    trend_periods=5,         # Periods for trend calculation
    trend_threshold=0.02     # Minimum change for trend
)
```

### Processors

#### DataPreprocessor
```python
preprocessor = DataPreprocessor(
    scaling_method='standard',      # 'standard', 'minmax', 'robust'
    imputation_method='forward_fill', # Missing value handling
    outlier_method='iqr'            # 'iqr', 'zscore', 'isolation_forest'
)
```

#### FeatureEngineer
```python
engineer = FeatureEngineer(
    include_technical=True,      # Technical indicators
    include_microstructure=True, # Market microstructure
    include_statistical=True,    # Statistical features
    include_rolling=True        # Rolling window features
)
```

### Strategies

#### MomentumStrategy
```python
strategy = MomentumStrategy(
    lookback_period=20,        # Momentum calculation period
    momentum_threshold=0.05,   # Minimum momentum for signal
    risk_per_trade=0.02,      # Risk per trade (2%)
    max_positions=5           # Maximum concurrent positions
)
```

### Utils

#### Backtester
```python
backtester = Backtester(
    initial_capital=100000,    # Starting capital
    commission=0.001,          # Commission rate
    slippage=0.0001,          # Slippage rate
    position_size=0.1         # Position size (10% of capital)
)
```

## Performance Optimization

1. **GPU Acceleration**: Enable GPU support for TensorFlow/PyTorch models
2. **Parallel Processing**: Use multiprocessing for feature engineering
3. **Caching**: Implement data caching to reduce API calls
4. **Batch Processing**: Process multiple symbols simultaneously

## Best Practices

1. **Data Quality**: Always check for missing values and outliers
2. **Feature Selection**: Use importance scores to select relevant features
3. **Cross-Validation**: Use time series cross-validation for model evaluation
4. **Risk Management**: Never risk more than 2% per trade
5. **Model Updates**: Retrain models regularly with new data

## Deployment

### API Deployment

```python
# Generate FastAPI deployment
from src.utils import ModelManager

manager = ModelManager()
deployment = manager.deploy_model(
    model_name='price_predictor',
    deployment_type='api'
)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or sequence length
2. **GPU Not Found**: Install CUDA and appropriate TensorFlow/PyTorch version
3. **API Rate Limits**: Implement caching and rate limiting
4. **Model Overfitting**: Add regularization, reduce model complexity

### Performance Tips

1. Use feature importance to reduce dimensionality
2. Implement early stopping in training
3. Use efficient data structures (numpy arrays)
4. Profile code to identify bottlenecks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is part of the FinTech Terminal and follows the same license terms.

## Support

For issues and questions:
- Check the documentation
- Look at example notebooks
- Open an issue on GitHub
- Contact the development team

## Roadmap

- [ ] Reinforcement learning for trading
- [ ] Options pricing models
- [ ] Portfolio optimization
- [ ] Real-time streaming predictions
- [ ] AutoML capabilities
- [ ] Quantum computing integration

## Acknowledgments

- TensorFlow/Keras for deep learning
- Scikit-learn for machine learning
- TA-Lib for technical analysis
- NewsAPI for news data
- The open-source community