"""
Simplified FastAPI backend with real-time market data
No PostgreSQL required - uses SQLite for simplicity
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep track of WebSocket connections
active_connections: List[WebSocket] = []

# Popular stocks to track
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
MARKET_INDICES = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow Jones, Nasdaq, VIX

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting market data streaming...")
    task = asyncio.create_task(stream_market_data())
    yield
    # Shutdown
    logger.info("Shutting down...")
    task.cancel()

app = FastAPI(title="Fintech Terminal API", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Fintech Terminal API",
        "status": "running",
        "endpoints": {
            "quotes": "/api/v1/market/quotes",
            "quote": "/api/v1/market/quote/{symbol}",
            "history": "/api/v1/market/history/{symbol}",
            "search": "/api/v1/market/search",
            "indices": "/api/v1/market/indices",
            "websocket": "ws://localhost:8000/ws/market"
        }
    }

@app.get("/api/v1/market/quotes")
async def get_multiple_quotes(symbols: str = None):
    """Get real-time quotes for multiple symbols"""
    try:
        symbol_list = symbols.split(",") if symbols else DEFAULT_SYMBOLS
        symbol_list = [s.strip().upper() for s in symbol_list]
        
        quotes = []
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period="1d", interval="1m")
                
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    open_price = history['Open'].iloc[0]
                    change = current_price - open_price
                    change_percent = (change / open_price) * 100 if open_price > 0 else 0
                    
                    quotes.append({
                        "symbol": symbol,
                        "name": info.get('longName', symbol),
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "changePercent": round(change_percent, 2),
                        "volume": int(history['Volume'].sum()),
                        "marketCap": info.get('marketCap', 0),
                        "dayHigh": round(history['High'].max(), 2),
                        "dayLow": round(history['Low'].min(), 2),
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                
        return {"quotes": quotes, "count": len(quotes)}
    except Exception as e:
        logger.error(f"Error in get_multiple_quotes: {e}")
        return {"quotes": [], "error": str(e)}

@app.get("/api/v1/market/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a single symbol"""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        history = ticker.history(period="1d", interval="1m")
        
        if history.empty:
            return {"error": f"No data found for {symbol}"}
            
        current_price = history['Close'].iloc[-1]
        open_price = history['Open'].iloc[0]
        change = current_price - open_price
        change_percent = (change / open_price) * 100 if open_price > 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "name": info.get('longName', symbol),
            "price": round(current_price, 2),
            "change": round(change, 2),
            "changePercent": round(change_percent, 2),
            "open": round(open_price, 2),
            "high": round(history['High'].max(), 2),
            "low": round(history['Low'].min(), 2),
            "volume": int(history['Volume'].sum()),
            "marketCap": info.get('marketCap', 0),
            "pe": info.get('trailingPE', 0),
            "eps": info.get('trailingEps', 0),
            "beta": info.get('beta', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}

@app.get("/api/v1/market/history/{symbol}")
async def get_history(symbol: str, period: str = "1mo", interval: str = "1d"):
    """Get historical data for a symbol"""
    try:
        ticker = yf.Ticker(symbol.upper())
        history = ticker.history(period=period, interval=interval)
        
        if history.empty:
            return {"error": f"No historical data found for {symbol}"}
            
        # Convert to list of dictionaries for JSON serialization
        data = []
        for index, row in history.iterrows():
            data.append({
                "timestamp": index.isoformat(),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
            
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}

@app.get("/api/v1/market/search")
async def search_symbols(query: str):
    """Search for stock symbols"""
    try:
        # For simplicity, using a predefined list
        # In production, you'd use a proper symbol database
        all_symbols = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla, Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson",
            "V": "Visa Inc.",
            "PG": "Procter & Gamble Co.",
            "UNH": "UnitedHealth Group Inc.",
            "HD": "The Home Depot Inc.",
            "MA": "Mastercard Inc.",
            "DIS": "The Walt Disney Company",
            "PYPL": "PayPal Holdings Inc.",
            "BAC": "Bank of America Corp.",
            "NFLX": "Netflix Inc.",
            "ADBE": "Adobe Inc.",
            "CRM": "Salesforce.com Inc."
        }
        
        query = query.upper()
        results = []
        
        for symbol, name in all_symbols.items():
            if query in symbol or query in name.upper():
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "type": "stock"
                })
                
        return {"results": results[:10]}  # Limit to 10 results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return {"results": [], "error": str(e)}

@app.get("/api/v1/market/indices")
async def get_market_indices():
    """Get major market indices"""
    try:
        indices = []
        for symbol in MARKET_INDICES:
            try:
                ticker = yf.Ticker(symbol)
                history = ticker.history(period="1d", interval="1m")
                
                if not history.empty:
                    current = history['Close'].iloc[-1]
                    open_price = history['Open'].iloc[0]
                    change = current - open_price
                    change_percent = (change / open_price) * 100 if open_price > 0 else 0
                    
                    name_map = {
                        "^GSPC": "S&P 500",
                        "^DJI": "Dow Jones",
                        "^IXIC": "NASDAQ",
                        "^VIX": "VIX"
                    }
                    
                    indices.append({
                        "symbol": symbol,
                        "name": name_map.get(symbol, symbol),
                        "value": round(current, 2),
                        "change": round(change, 2),
                        "changePercent": round(change_percent, 2)
                    })
            except Exception as e:
                logger.error(f"Error fetching index {symbol}: {e}")
                
        return {"indices": indices}
    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        return {"indices": [], "error": str(e)}

@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(active_connections)}")

async def stream_market_data():
    """Stream real-time market data to all connected clients"""
    while True:
        try:
            if active_connections:
                # Fetch latest data for default symbols
                quotes = []
                for symbol in DEFAULT_SYMBOLS[:5]:  # Limit to 5 symbols for performance
                    try:
                        ticker = yf.Ticker(symbol)
                        history = ticker.history(period="1d", interval="1m")
                        
                        if not history.empty:
                            current_price = history['Close'].iloc[-1]
                            open_price = history['Open'].iloc[0]
                            change = current_price - open_price
                            change_percent = (change / open_price) * 100 if open_price > 0 else 0
                            
                            quotes.append({
                                "symbol": symbol,
                                "price": round(current_price, 2),
                                "change": round(change, 2),
                                "changePercent": round(change_percent, 2),
                                "volume": int(history['Volume'].iloc[-1]),
                                "timestamp": datetime.now().isoformat()
                            })
                    except Exception as e:
                        logger.error(f"Error streaming {symbol}: {e}")
                
                # Send to all connected clients
                message = json.dumps({
                    "type": "market_update",
                    "data": quotes,
                    "timestamp": datetime.now().isoformat()
                })
                
                disconnected = []
                for connection in active_connections:
                    try:
                        await connection.send_text(message)
                    except Exception:
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for conn in disconnected:
                    active_connections.remove(conn)
                    
            # Wait before next update (5 seconds for free tier rate limits)
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in stream_market_data: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)