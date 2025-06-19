"""
Enhanced real-time market data service with multiple API sources
"""
import asyncio
import aiohttp
import yfinance as yf
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class RealTimeMarketService:
    """
    Service for real-time market data with fallback APIs
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.session: Optional[aiohttp.ClientSession] = None
        self.subscribed_symbols: Set[str] = set()
        self.last_prices: Dict[str, float] = {}
        self.api_keys = {
            'finnhub': 'demo',  # Replace with your Finnhub API key
            'alpha_vantage': 'demo',  # Replace with your Alpha Vantage API key
        }
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            
    def _fetch_yahoo_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance using yfinance"""
        try:
            # Join symbols for batch request
            tickers = yf.Tickers(' '.join(symbols))
            
            results = {}
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        price = float(latest['Close'])
                        
                        # Calculate change from previous close
                        prev_close = info.get('regularMarketPreviousClose', price)
                        change = price - prev_close
                        change_percent = (change / prev_close * 100) if prev_close else 0
                        
                        results[symbol] = {
                            'symbol': symbol,
                            'price': price,
                            'change': change,
                            'changePercent': change_percent,
                            'volume': int(latest['Volume']),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'yahoo'
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol} from Yahoo: {e}")
                    continue
                    
            return results
        except Exception as e:
            logger.error(f"Yahoo Finance batch request failed: {e}")
            return {}
    
    async def _fetch_finnhub_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch data from Finnhub API"""
        if self.api_keys['finnhub'] == 'demo':
            return {}  # Skip if no API key
            
        session = await self._get_session()
        results = {}
        
        try:
            for symbol in symbols:
                url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_keys['finnhub']}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'c' in data:  # Current price
                            price = float(data['c'])
                            prev_close = float(data['pc'])  # Previous close
                            change = price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close else 0
                            
                            results[symbol] = {
                                'symbol': symbol,
                                'price': price,
                                'change': change,
                                'changePercent': change_percent,
                                'volume': int(data.get('v', 0)),  # Volume
                                'timestamp': datetime.now().isoformat(),
                                'source': 'finnhub'
                            }
                
                # Rate limiting for free tier (60 calls/minute)
                await asyncio.sleep(1.1)
                
        except Exception as e:
            logger.error(f"Finnhub API request failed: {e}")
            
        return results
    
    async def get_real_time_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get real-time quotes with fallback APIs
        """
        all_results = {}
        
        # Try Yahoo Finance first (fastest and most reliable for demo)
        try:
            yahoo_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._fetch_yahoo_data, symbols
            )
            all_results.update(yahoo_results)
        except Exception as e:
            logger.error(f"Yahoo Finance failed: {e}")
        
        # Fill missing symbols with Finnhub (if API key available)
        missing_symbols = [s for s in symbols if s not in all_results]
        if missing_symbols and self.api_keys['finnhub'] != 'demo':
            try:
                finnhub_results = await self._fetch_finnhub_data(missing_symbols)
                all_results.update(finnhub_results)
            except Exception as e:
                logger.error(f"Finnhub fallback failed: {e}")
        
        # Update last known prices
        for symbol, data in all_results.items():
            self.last_prices[symbol] = data['price']
        
        return list(all_results.values())
    
    async def start_real_time_updates(
        self, 
        symbols: List[str], 
        callback: callable, 
        interval_seconds: int = 5
    ):
        """
        Start real-time updates for given symbols
        """
        self.subscribed_symbols.update(symbols)
        
        async def update_loop():
            while self.subscribed_symbols:
                try:
                    # Get current subscribed symbols
                    current_symbols = list(self.subscribed_symbols)
                    
                    if current_symbols:
                        # Fetch updates
                        updates = await self.get_real_time_quotes(current_symbols)
                        
                        # Send updates via callback
                        if updates:
                            await callback({
                                'type': 'market_update',
                                'data': updates,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Wait for next update
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Real-time update loop error: {e}")
                    await asyncio.sleep(interval_seconds)
        
        # Start the update loop
        asyncio.create_task(update_loop())
    
    def subscribe_symbol(self, symbol: str):
        """Subscribe to a symbol for real-time updates"""
        self.subscribed_symbols.add(symbol.upper())
    
    def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol"""
        self.subscribed_symbols.discard(symbol.upper())
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols"""
        return list(self.subscribed_symbols)

# Global instance
realtime_service = RealTimeMarketService()