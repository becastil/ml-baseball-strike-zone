"""
Minimal FastAPI app for real-time market data - no database required
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Fintech Terminal API - Minimal", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections: List[WebSocket] = []

@app.get("/")
async def root():
    return {"message": "Fintech Terminal API - Minimal Version"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Fintech Terminal API",
        "version": "0.1.0"
    }

@app.websocket("/api/v1/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to real-time market data",
            "timestamp": datetime.now().isoformat()
        })
        
        # Start sending market updates
        asyncio.create_task(send_market_updates(websocket))
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            elif message.get("action") == "subscribe":
                symbols = message.get("symbols", [])
                await websocket.send_json({
                    "type": "subscribed",
                    "symbols": symbols,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def send_market_updates(websocket: WebSocket):
    """Send market updates every 5 seconds"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    
    while websocket in active_connections:
        try:
            # Fetch market data
            market_data = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        price = float(latest['Close'])
                        prev_close = info.get('regularMarketPreviousClose', price)
                        change = price - prev_close
                        change_percent = (change / prev_close * 100) if prev_close else 0
                        
                        market_data.append({
                            "symbol": symbol,
                            "price": round(price, 2),
                            "change": round(change, 2),
                            "changePercent": round(change_percent, 2),
                            "volume": int(latest['Volume']),
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
                    continue
            
            # Send update
            if market_data:
                await websocket.send_json({
                    "type": "market_update",
                    "data": market_data,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error sending market updates: {e}")
            break

@app.get("/api/v1/realtime/status")
async def get_status():
    return {
        "active_connections": len(active_connections),
        "service_status": "running",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Minimal Fintech Terminal API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)