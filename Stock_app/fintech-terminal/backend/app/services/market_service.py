"""
Market data fetching service
"""
import yfinance as yf
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.schemas.market import (
    StockQuote,
    HistoricalData,
    MarketOverview,
    SearchResult,
    CompanyInfo,
    PriceData
)


class MarketService:
    """
    Service for fetching market data from various sources
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    def _fetch_yfinance_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch quote data using yfinance (blocking)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                # Try to get basic quote
                hist = ticker.history(period="1d")
                if hist.empty:
                    return None
                
                last_close = hist['Close'].iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(last_close),
                    'name': symbol,
                    'exchange': 'Unknown',
                    'currency': 'USD'
                }
            
            return {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', symbol)),
                'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'previousClose': info.get('regularMarketPreviousClose', 0),
                'open': info.get('regularMarketOpen', 0),
                'dayLow': info.get('regularMarketDayLow', 0),
                'dayHigh': info.get('regularMarketDayHigh', 0),
                'volume': info.get('regularMarketVolume', 0),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', None),
                'eps': info.get('trailingEps', None),
                'beta': info.get('beta', None),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'dividendYield': info.get('dividendYield', None),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'quoteType': info.get('quoteType', 'EQUITY')
            }
        except Exception as e:
            print(f"Error fetching quote for {symbol}: {e}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get real-time quote for a single symbol
        """
        loop = asyncio.get_event_loop()
        quote_data = await loop.run_in_executor(
            self.executor,
            self._fetch_yfinance_quote,
            symbol
        )
        
        if not quote_data:
            return None
        
        # Calculate change and change percentage
        price = quote_data.get('price', 0)
        prev_close = quote_data.get('previousClose', price)
        change = price - prev_close if prev_close else 0
        change_percent = (change / prev_close * 100) if prev_close and prev_close != 0 else 0
        
        return StockQuote(
            symbol=quote_data['symbol'],
            name=quote_data['name'],
            price=price,
            change=change,
            changePercent=change_percent,
            volume=quote_data.get('volume', 0),
            marketCap=quote_data.get('marketCap', 0),
            previousClose=prev_close,
            open=quote_data.get('open', 0),
            dayHigh=quote_data.get('dayHigh', 0),
            dayLow=quote_data.get('dayLow', 0),
            fiftyTwoWeekHigh=quote_data.get('fiftyTwoWeekHigh', 0),
            fiftyTwoWeekLow=quote_data.get('fiftyTwoWeekLow', 0),
            peRatio=quote_data.get('peRatio'),
            eps=quote_data.get('eps'),
            beta=quote_data.get('beta'),
            dividendYield=quote_data.get('dividendYield'),
            exchange=quote_data.get('exchange', 'Unknown'),
            currency=quote_data.get('currency', 'USD'),
            timestamp=datetime.utcnow()
        )
    
    async def get_multiple_quotes(self, symbols: List[str]) -> List[StockQuote]:
        """
        Get quotes for multiple symbols
        """
        tasks = [self.get_quote(symbol) for symbol in symbols]
        quotes = await asyncio.gather(*tasks)
        return [q for q in quotes if q is not None]
    
    def _fetch_yfinance_historical(
        self, 
        symbol: str, 
        period: str, 
        interval: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical data using yfinance (blocking)
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return None
            
            # Convert to list of price data
            prices = []
            for index, row in hist.iterrows():
                prices.append({
                    'timestamp': index,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return {
                'symbol': symbol,
                'prices': prices,
                'period': period,
                'interval': interval
            }
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ) -> Optional[HistoricalData]:
        """
        Get historical price data
        """
        loop = asyncio.get_event_loop()
        hist_data = await loop.run_in_executor(
            self.executor,
            self._fetch_yfinance_historical,
            symbol,
            period,
            interval
        )
        
        if not hist_data:
            return None
        
        prices = [
            PriceData(
                timestamp=p['timestamp'],
                open=p['open'],
                high=p['high'],
                low=p['low'],
                close=p['close'],
                volume=p['volume']
            )
            for p in hist_data['prices']
        ]
        
        return HistoricalData(
            symbol=symbol,
            prices=prices,
            period=period,
            interval=interval
        )
    
    async def search_symbols(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for symbols by query
        """
        # This is a simplified implementation
        # In production, you'd use a proper symbol search API
        
        # For now, we'll return some mock results
        # You could integrate with Alpha Vantage, IEX Cloud, or other APIs
        
        mock_results = [
            SearchResult(
                symbol="AAPL",
                name="Apple Inc.",
                type="Stock",
                exchange="NASDAQ",
                currency="USD"
            ),
            SearchResult(
                symbol="GOOGL",
                name="Alphabet Inc.",
                type="Stock",
                exchange="NASDAQ",
                currency="USD"
            )
        ]
        
        # Filter by query
        results = [
            r for r in mock_results 
            if query.upper() in r.symbol or query.lower() in r.name.lower()
        ]
        
        return results[:limit]
    
    def _fetch_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company information using yfinance (blocking)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'description': info.get('longBusinessSummary'),
                'website': info.get('website'),
                'employees': info.get('fullTimeEmployees'),
                'headquarters': f"{info.get('city', '')}, {info.get('state', '')} {info.get('country', '')}".strip(),
                'founded': info.get('founded'),
                'ceo': info.get('companyOfficers', [{}])[0].get('name') if info.get('companyOfficers') else None
            }
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return None
    
    async def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """
        Get detailed company information
        """
        loop = asyncio.get_event_loop()
        company_data = await loop.run_in_executor(
            self.executor,
            self._fetch_company_info,
            symbol
        )
        
        if not company_data:
            return None
        
        return CompanyInfo(
            symbol=company_data['symbol'],
            name=company_data['name'],
            sector=company_data.get('sector'),
            industry=company_data.get('industry'),
            description=company_data.get('description'),
            website=company_data.get('website'),
            employees=company_data.get('employees'),
            headquarters=company_data.get('headquarters'),
            founded=company_data.get('founded'),
            ceo=company_data.get('ceo')
        )
    
    async def get_market_overview(self) -> MarketOverview:
        """
        Get market overview with major indices
        """
        # Define major market indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        
        # Fetch quotes for indices
        index_quotes = await self.get_multiple_quotes(indices)
        
        # Determine if market is open (simplified)
        now = datetime.utcnow()
        weekday = now.weekday()
        hour = now.hour
        
        # Market is open Monday-Friday, 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
        is_open = weekday < 5 and 14 <= hour < 21
        
        return MarketOverview(
            marketStatus="open" if is_open else "closed",
            indices=[
                {
                    "symbol": q.symbol,
                    "name": q.name,
                    "value": q.price,
                    "change": q.change,
                    "changePercent": q.changePercent
                }
                for q in index_quotes
            ],
            timestamp=datetime.utcnow()
        )