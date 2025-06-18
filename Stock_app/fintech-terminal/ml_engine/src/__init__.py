"""
FinTech Terminal ML Engine

A comprehensive machine learning and AI module for financial analysis,
prediction, and trading strategy development.
"""

__version__ = "1.0.0"
__author__ = "FinTech Terminal Team"

# Import main components for easier access
from .predictors import PricePredictor, TrendClassifier
from .processors import DataPreprocessor, FeatureEngineer
from .analysis import TechnicalAnalysis, SentimentAnalyzer
from .strategies import MomentumStrategy
from .utils import ModelManager, Backtester

__all__ = [
    "PricePredictor",
    "TrendClassifier",
    "DataPreprocessor",
    "FeatureEngineer",
    "TechnicalAnalysis",
    "SentimentAnalyzer",
    "MomentumStrategy",
    "ModelManager",
    "Backtester"
]