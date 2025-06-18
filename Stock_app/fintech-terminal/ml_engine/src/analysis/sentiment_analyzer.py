"""
Sentiment Analysis Module

This module analyzes market sentiment from news articles, social media,
and other text sources to generate trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re
from datetime import datetime, timedelta
import requests
import json
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class SentimentAnalyzer:
    """Comprehensive sentiment analysis for financial markets"""
    
    def __init__(self,
                 use_vader: bool = True,
                 use_textblob: bool = True,
                 use_finbert: bool = False,
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the sentiment analyzer
        
        Args:
            use_vader: Use VADER sentiment analyzer
            use_textblob: Use TextBlob sentiment analyzer
            use_finbert: Use FinBERT for financial sentiment
            api_keys: Dictionary of API keys for news sources
        """
        self.use_vader = use_vader
        self.use_textblob = use_textblob
        self.use_finbert = use_finbert
        self.api_keys = api_keys or {}
        
        # Initialize sentiment analyzers
        if self.use_vader:
            self.vader = SentimentIntensityAnalyzer()
        
        if self.use_finbert:
            try:
                self.finbert = pipeline("sentiment-analysis", 
                                      model="ProsusAI/finbert",
                                      device=-1)  # CPU
            except:
                print("FinBERT not available, using standard analyzers")
                self.use_finbert = False
        
        self.sentiment_history = pd.DataFrame()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        results = {}
        
        # Clean text
        text = self._preprocess_text(text)
        
        # VADER sentiment
        if self.use_vader:
            vader_scores = self.vader.polarity_scores(text)
            results['vader_compound'] = vader_scores['compound']
            results['vader_positive'] = vader_scores['pos']
            results['vader_negative'] = vader_scores['neg']
            results['vader_neutral'] = vader_scores['neu']
        
        # TextBlob sentiment
        if self.use_textblob:
            blob = TextBlob(text)
            results['textblob_polarity'] = blob.sentiment.polarity
            results['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # FinBERT sentiment
        if self.use_finbert and len(text) > 10:
            try:
                # FinBERT has max length, so truncate if needed
                finbert_result = self.finbert(text[:512])[0]
                label = finbert_result['label'].lower()
                score = finbert_result['score']
                
                if label == 'positive':
                    results['finbert_sentiment'] = score
                elif label == 'negative':
                    results['finbert_sentiment'] = -score
                else:  # neutral
                    results['finbert_sentiment'] = 0
            except:
                results['finbert_sentiment'] = 0
        
        # Combined sentiment score
        sentiment_scores = []
        if 'vader_compound' in results:
            sentiment_scores.append(results['vader_compound'])
        if 'textblob_polarity' in results:
            sentiment_scores.append(results['textblob_polarity'])
        if 'finbert_sentiment' in results:
            sentiment_scores.append(results['finbert_sentiment'])
        
        if sentiment_scores:
            results['combined_sentiment'] = np.mean(sentiment_scores)
        else:
            results['combined_sentiment'] = 0
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_news(self, 
                    symbol: str,
                    days_back: int = 7,
                    sources: List[str] = ['newsapi', 'reddit']) -> pd.DataFrame:
        """
        Analyze sentiment from news sources
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            sources: List of news sources to use
            
        Returns:
            DataFrame with sentiment analysis results
        """
        all_articles = []
        
        # Fetch news from different sources
        if 'newsapi' in sources and 'newsapi' in self.api_keys:
            articles = self._fetch_newsapi(symbol, days_back)
            all_articles.extend(articles)
        
        if 'reddit' in sources:
            posts = self._fetch_reddit(symbol, days_back)
            all_articles.extend(posts)
        
        # Analyze sentiment for each article
        results = []
        for article in all_articles:
            sentiment = self.analyze_text(article['text'])
            
            result = {
                'timestamp': article['timestamp'],
                'source': article['source'],
                'title': article.get('title', ''),
                **sentiment
            }
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return df
    
    def _fetch_newsapi(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        articles = []
        
        try:
            api_key = self.api_keys.get('newsapi')
            if not api_key:
                return articles
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Build query
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f'{symbol} stock',
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    articles.append({
                        'timestamp': article['publishedAt'],
                        'source': 'newsapi',
                        'title': article['title'],
                        'text': f"{article['title']} {article.get('description', '')}"
                    })
        except Exception as e:
            print(f"Error fetching NewsAPI: {e}")
        
        return articles
    
    def _fetch_reddit(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch posts from Reddit (simplified, no API key needed)"""
        posts = []
        
        try:
            # Subreddits to search
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            
            for subreddit in subreddits:
                url = f'https://www.reddit.com/r/{subreddit}/search.json'
                params = {
                    'q': symbol,
                    'sort': 'new',
                    'limit': 25,
                    't': 'week'
                }
                
                headers = {'User-Agent': 'FinTech Terminal 1.0'}
                response = requests.get(url, params=params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data['data']['children']:
                        post_data = post['data']
                        created_time = datetime.fromtimestamp(post_data['created_utc'])
                        
                        # Check if within date range
                        if created_time >= datetime.now() - timedelta(days=days_back):
                            posts.append({
                                'timestamp': created_time,
                                'source': f'reddit/{subreddit}',
                                'title': post_data['title'],
                                'text': f"{post_data['title']} {post_data.get('selftext', '')}"
                            })
        except Exception as e:
            print(f"Error fetching Reddit: {e}")
        
        return posts
    
    def calculate_sentiment_indicators(self, 
                                     sentiment_df: pd.DataFrame,
                                     price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment-based trading indicators
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            price_df: DataFrame with price data
            
        Returns:
            DataFrame with sentiment indicators
        """
        # Resample sentiment to daily frequency
        if not sentiment_df.empty:
            daily_sentiment = sentiment_df.set_index('timestamp').resample('D').agg({
                'combined_sentiment': ['mean', 'std', 'count'],
                'vader_compound': 'mean',
                'textblob_polarity': 'mean'
            })
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns]
        else:
            # Create empty sentiment data
            daily_sentiment = pd.DataFrame(index=price_df.index)
            daily_sentiment['combined_sentiment_mean'] = 0
            daily_sentiment['combined_sentiment_std'] = 0
            daily_sentiment['combined_sentiment_count'] = 0
        
        # Merge with price data
        merged = price_df.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        merged.fillna(method='ffill', inplace=True)
        
        # Calculate sentiment indicators
        # Moving average of sentiment
        merged['sentiment_ma_7'] = merged['combined_sentiment_mean'].rolling(window=7).mean()
        merged['sentiment_ma_30'] = merged['combined_sentiment_mean'].rolling(window=30).mean()
        
        # Sentiment momentum
        merged['sentiment_momentum'] = merged['combined_sentiment_mean'].diff(5)
        
        # Sentiment volatility
        merged['sentiment_volatility'] = merged['combined_sentiment_mean'].rolling(window=20).std()
        
        # Sentiment extremes
        merged['sentiment_percentile'] = merged['combined_sentiment_mean'].rolling(window=100).rank(pct=True)
        
        # Volume of mentions (news volume)
        merged['news_volume'] = merged['combined_sentiment_count'].fillna(0)
        merged['news_volume_ma'] = merged['news_volume'].rolling(window=7).mean()
        
        # Sentiment-price divergence
        price_change = merged['close'].pct_change(5)
        sentiment_change = merged['combined_sentiment_mean'].diff(5)
        merged['sentiment_price_divergence'] = sentiment_change - price_change
        
        # Sentiment regime
        merged['sentiment_regime'] = pd.cut(merged['combined_sentiment_mean'],
                                           bins=[-1, -0.3, 0.3, 1],
                                           labels=['bearish', 'neutral', 'bullish'])
        
        return merged
    
    def generate_sentiment_signals(self,
                                 sentiment_indicators: pd.DataFrame,
                                 strategy: str = 'threshold') -> pd.Series:
        """
        Generate trading signals based on sentiment
        
        Args:
            sentiment_indicators: DataFrame with sentiment indicators
            strategy: Signal generation strategy
            
        Returns:
            Series with trading signals
        """
        signals = pd.Series(0, index=sentiment_indicators.index)
        
        if strategy == 'threshold':
            # Simple threshold-based signals
            signals[sentiment_indicators['combined_sentiment_mean'] > 0.5] = 1
            signals[sentiment_indicators['combined_sentiment_mean'] < -0.5] = -1
            
        elif strategy == 'momentum':
            # Sentiment momentum signals
            signals[sentiment_indicators['sentiment_momentum'] > 0.1] = 1
            signals[sentiment_indicators['sentiment_momentum'] < -0.1] = -1
            
        elif strategy == 'extreme':
            # Trade on sentiment extremes (contrarian)
            signals[sentiment_indicators['sentiment_percentile'] < 0.1] = 1  # Oversold sentiment
            signals[sentiment_indicators['sentiment_percentile'] > 0.9] = -1  # Overbought sentiment
            
        elif strategy == 'divergence':
            # Trade on sentiment-price divergence
            div = sentiment_indicators['sentiment_price_divergence']
            signals[div > div.rolling(20).std() * 2] = 1
            signals[div < -div.rolling(20).std() * 2] = -1
            
        elif strategy == 'combined':
            # Combine multiple sentiment signals
            # Threshold component
            threshold_signal = pd.Series(0, index=sentiment_indicators.index)
            threshold_signal[sentiment_indicators['combined_sentiment_mean'] > 0.3] = 1
            threshold_signal[sentiment_indicators['combined_sentiment_mean'] < -0.3] = -1
            
            # Momentum component
            momentum_signal = pd.Series(0, index=sentiment_indicators.index)
            momentum_signal[sentiment_indicators['sentiment_momentum'] > 0.05] = 1
            momentum_signal[sentiment_indicators['sentiment_momentum'] < -0.05] = -1
            
            # News volume component
            volume_signal = pd.Series(0, index=sentiment_indicators.index)
            high_volume = sentiment_indicators['news_volume'] > sentiment_indicators['news_volume_ma'] * 2
            volume_signal[high_volume & (sentiment_indicators['combined_sentiment_mean'] > 0)] = 1
            volume_signal[high_volume & (sentiment_indicators['combined_sentiment_mean'] < 0)] = -1
            
            # Combine signals
            combined = threshold_signal + momentum_signal + volume_signal
            signals[combined >= 2] = 1
            signals[combined <= -2] = -1
        
        return signals
    
    def calculate_sentiment_score(self, 
                                symbol: str,
                                lookback_days: int = 7) -> Dict[str, float]:
        """
        Calculate current sentiment score for a symbol
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        # Fetch and analyze recent news
        sentiment_df = self.analyze_news(symbol, lookback_days)
        
        if sentiment_df.empty:
            return {
                'current_sentiment': 0,
                'sentiment_trend': 0,
                'news_volume': 0,
                'sentiment_volatility': 0
            }
        
        # Calculate metrics
        recent_sentiment = sentiment_df['combined_sentiment'].tail(10).mean()
        older_sentiment = sentiment_df['combined_sentiment'].head(10).mean()
        
        return {
            'current_sentiment': recent_sentiment,
            'sentiment_trend': recent_sentiment - older_sentiment,
            'news_volume': len(sentiment_df),
            'sentiment_volatility': sentiment_df['combined_sentiment'].std(),
            'positive_ratio': (sentiment_df['combined_sentiment'] > 0).mean(),
            'average_sentiment': sentiment_df['combined_sentiment'].mean()
        }
    
    def create_sentiment_report(self, 
                              symbol: str,
                              sentiment_df: pd.DataFrame) -> Dict:
        """
        Create a comprehensive sentiment report
        
        Args:
            symbol: Stock symbol
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            Dictionary with sentiment report
        """
        if sentiment_df.empty:
            return {
                'symbol': symbol,
                'summary': 'No sentiment data available',
                'metrics': {},
                'top_positive': [],
                'top_negative': []
            }
        
        # Calculate summary metrics
        metrics = {
            'total_articles': len(sentiment_df),
            'average_sentiment': sentiment_df['combined_sentiment'].mean(),
            'sentiment_std': sentiment_df['combined_sentiment'].std(),
            'positive_articles': (sentiment_df['combined_sentiment'] > 0.1).sum(),
            'negative_articles': (sentiment_df['combined_sentiment'] < -0.1).sum(),
            'neutral_articles': ((sentiment_df['combined_sentiment'] >= -0.1) & 
                               (sentiment_df['combined_sentiment'] <= 0.1)).sum()
        }
        
        # Get top positive and negative articles
        top_positive = sentiment_df.nlargest(5, 'combined_sentiment')[
            ['timestamp', 'title', 'combined_sentiment', 'source']
        ].to_dict('records')
        
        top_negative = sentiment_df.nsmallest(5, 'combined_sentiment')[
            ['timestamp', 'title', 'combined_sentiment', 'source']
        ].to_dict('records')
        
        # Generate summary
        avg_sentiment = metrics['average_sentiment']
        if avg_sentiment > 0.3:
            summary = f"Very positive sentiment for {symbol}"
        elif avg_sentiment > 0.1:
            summary = f"Positive sentiment for {symbol}"
        elif avg_sentiment < -0.3:
            summary = f"Very negative sentiment for {symbol}"
        elif avg_sentiment < -0.1:
            summary = f"Negative sentiment for {symbol}"
        else:
            summary = f"Neutral sentiment for {symbol}"
        
        return {
            'symbol': symbol,
            'summary': summary,
            'metrics': metrics,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'sentiment_by_source': sentiment_df.groupby('source')['combined_sentiment'].mean().to_dict(),
            'daily_sentiment': sentiment_df.set_index('timestamp').resample('D')['combined_sentiment'].mean().to_dict()
        }