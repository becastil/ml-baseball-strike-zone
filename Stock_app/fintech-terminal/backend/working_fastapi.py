#!/usr/bin/env python3
"""
Working FastAPI Server - Minimal Dependencies
This is a fully functional FastAPI server that should work with Python 3.13
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Create FastAPI app
app = FastAPI(
    title="FinTech Terminal API",
    description="A working FastAPI server for the FinTech Terminal",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data storage
MOCK_STOCKS = {
    "AAPL": {"name": "Apple Inc.", "base_price": 150.25, "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc.", "base_price": 2750.30, "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corporation", "base_price": 310.40, "sector": "Technology"},
    "TSLA": {"name": "Tesla, Inc.", "base_price": 240.85, "sector": "Automotive"},
    "AMZN": {"name": "Amazon.com Inc.", "base_price": 3200.15, "sector": "E-commerce"},
    "NVDA": {"name": "NVIDIA Corporation", "base_price": 450.75, "sector": "Technology"},
    "META": {"name": "Meta Platforms, Inc.", "base_price": 280.60, "sector": "Technology"},
    "NFLX": {"name": "Netflix, Inc.", "base_price": 420.30, "sector": "Entertainment"}
}

def generate_mock_quote(symbol: str) -> Dict[str, Any]:
    """Generate realistic mock stock data"""
    if symbol not in MOCK_STOCKS:
        return None
    
    stock_info = MOCK_STOCKS[symbol]
    base_price = stock_info["base_price"]
    
    # Generate realistic price movement
    price_change = random.uniform(-base_price * 0.05, base_price * 0.05)  # ±5% max change
    current_price = base_price + price_change
    change_percent = (price_change / base_price) * 100
    
    return {
        "symbol": symbol,
        "name": stock_info["name"],
        "price": round(current_price, 2),
        "change": round(price_change, 2),
        "changePercent": round(change_percent, 2),
        "volume": random.randint(1000000, 100000000),
        "marketCap": random.randint(50000000000, 3000000000000),
        "previousClose": base_price,
        "open": round(base_price + random.uniform(-5, 5), 2),
        "dayHigh": round(current_price + random.uniform(0, 10), 2),
        "dayLow": round(current_price - random.uniform(0, 10), 2),
        "fiftyTwoWeekHigh": round(base_price * 1.3, 2),
        "fiftyTwoWeekLow": round(base_price * 0.7, 2),
        "peRatio": round(random.uniform(15, 35), 2),
        "eps": round(random.uniform(5, 25), 2),
        "beta": round(random.uniform(0.5, 2.0), 2),
        "dividendYield": round(random.uniform(0, 3), 2),
        "exchange": "NASDAQ",
        "currency": "USD",
        "sector": stock_info["sector"],
        "timestamp": datetime.now().isoformat(),
        "source": "mock_data"
    }

# API Routes
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 FinTech Terminal API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "market_quotes": "/api/v1/market/quotes?symbols=AAPL,GOOGL,MSFT",
            "single_quote": "/api/v1/market/quote/AAPL",
            "search": "/api/v1/market/search?query=apple",
            "market_overview": "/api/v1/market/overview"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FinTech Terminal API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": ["market_data", "search", "real_time_mock"]
    }

@app.get("/api/v1/market/quote/{symbol}")
def get_quote(symbol: str):
    """Get quote for a single symbol"""
    symbol = symbol.upper()
    quote = generate_mock_quote(symbol)
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    
    return quote

@app.get("/api/v1/market/quotes")
def get_multiple_quotes(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Get quotes for multiple symbols"""
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbol_list) > 50:
        raise HTTPException(status_code=400, detail="Too many symbols (max 50)")
    
    quotes = []
    for symbol in symbol_list:
        quote = generate_mock_quote(symbol)
        if quote:
            quotes.append(quote)
    
    return {
        "quotes": quotes,
        "count": len(quotes),
        "requested": len(symbol_list),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/market/search")
def search_symbols(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """Search for symbols by name or symbol"""
    query = query.lower()
    
    results = []
    for symbol, info in MOCK_STOCKS.items():
        if (query in symbol.lower() or 
            query in info["name"].lower() or 
            query in info["sector"].lower()):
            results.append({
                "symbol": symbol,
                "name": info["name"],
                "type": "Stock",
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": info["sector"]
            })
    
    # Limit results
    results = results[:limit]
    
    return {
        "results": results,
        "count": len(results),
        "query": query,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/market/overview")
def get_market_overview():
    """Get market overview with major indices"""
    # Mock major indices
    indices = [
        {"symbol": "^GSPC", "name": "S&P 500", "value": 4200 + random.uniform(-50, 50)},
        {"symbol": "^DJI", "name": "Dow Jones", "value": 33000 + random.uniform(-500, 500)},
        {"symbol": "^IXIC", "name": "NASDAQ", "value": 13000 + random.uniform(-200, 200)},
        {"symbol": "^RUT", "name": "Russell 2000", "value": 1800 + random.uniform(-30, 30)}
    ]
    
    for index in indices:
        change = random.uniform(-2, 2)
        index["change"] = round(change, 2)
        index["changePercent"] = round((change / index["value"]) * 100, 2)
        index["value"] = round(index["value"], 2)
    
    # Determine if market is open (simplified)
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()
    is_open = weekday < 5 and 9 <= hour <= 16  # Simplified market hours
    
    return {
        "marketStatus": "open" if is_open else "closed",
        "indices": indices,
        "timestamp": datetime.now().isoformat(),
        "note": "This is mock data for testing purposes"
    }

@app.get("/api/v1/realtime/status")
def get_realtime_status():
    """Get real-time service status"""
    return {
        "service_status": "running",
        "active_connections": 0,
        "websocket_available": False,
        "message": "Mock data server - WebSocket not implemented",
        "polling_recommended": True,
        "poll_interval_seconds": 5,
        "timestamp": datetime.now().isoformat()
    }

# Optional: Add a simple polling endpoint for real-time-like updates
@app.get("/api/v1/realtime/poll")
def poll_market_data(symbols: str = Query("AAPL,GOOGL,MSFT,TSLA", description="Symbols to poll")):
    """Polling endpoint for real-time-like updates"""
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    quotes = []
    for symbol in symbol_list:
        quote = generate_mock_quote(symbol)
        if quote:
            quotes.append(quote)
    
    return {
        "type": "market_update",
        "data": quotes,
        "timestamp": datetime.now().isoformat(),
        "source": "polling",
        "next_poll_in_seconds": 5
    }

def main():
    """Run the server"""
    print("🚀 Starting FinTech Terminal FastAPI Server")
    print("=" * 50)
    print("📊 Features:")
    print("  ✅ FastAPI with automatic docs")
    print("  ✅ CORS enabled for frontend")
    print("  ✅ Mock market data")
    print("  ✅ Real-time polling endpoint")
    print("  ✅ Symbol search")
    print("  ✅ Market overview")
    print()
    print("🌐 Endpoints:")
    print("  • API Docs: http://localhost:8000/docs")
    print("  • Health: http://localhost:8000/health")
    print("  • Quotes: http://localhost:8000/api/v1/market/quotes?symbols=AAPL,MSFT")
    print("  • Search: http://localhost:8000/api/v1/market/search?query=apple")
    print("  • Overview: http://localhost:8000/api/v1/market/overview")
    print("  • Polling: http://localhost:8000/api/v1/realtime/poll")
    print()
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()