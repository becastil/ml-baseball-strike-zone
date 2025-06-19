"""
Real-time market data API endpoints with WebSocket support
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import List, Dict, Any, Set
from datetime import datetime
import asyncio
import json
import logging

from app.services.realtime_service import realtime_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            # Unsubscribe from all symbols for this connection
            symbols = self.subscriptions[websocket]
            for symbol in symbols:
                # Check if any other connection is subscribed to this symbol
                still_needed = any(
                    symbol in subs for ws, subs in self.subscriptions.items() 
                    if ws != websocket
                )
                if not still_needed:
                    realtime_service.unsubscribe_symbol(symbol)
            
            del self.subscriptions[websocket]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def get_all_subscribed_symbols(self) -> Set[str]:
        """Get all symbols that are currently subscribed by any client"""
        all_symbols = set()
        for symbols in self.subscriptions.values():
            all_symbols.update(symbols)
        return all_symbols

# Global connection manager
manager = ConnectionManager()


async def broadcast_market_updates(update_data: Dict[str, Any]):
    """Callback function to broadcast market updates to all connected clients"""
    if manager.active_connections:
        await manager.broadcast(update_data)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data
    
    Message format:
    - Subscribe: {"action": "subscribe", "symbols": ["AAPL", "GOOGL"]}
    - Unsubscribe: {"action": "unsubscribe", "symbols": ["AAPL"]}
    - Ping: {"action": "ping"}
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            'type': 'welcome',
            'message': 'Connected to real-time market data',
            'timestamp': datetime.now().isoformat()
        }, websocket)
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get('action')
            
            if action == 'subscribe':
                symbols = [s.upper() for s in message.get('symbols', [])]
                
                # Add to this connection's subscriptions
                manager.subscriptions[websocket].update(symbols)
                
                # Subscribe to real-time service
                for symbol in symbols:
                    realtime_service.subscribe_symbol(symbol)
                
                # Send immediate data for subscribed symbols
                if symbols:
                    try:
                        quotes = await realtime_service.get_real_time_quotes(symbols)
                        await manager.send_personal_message({
                            'type': 'initial_data',
                            'data': quotes,
                            'symbols': symbols,
                            'timestamp': datetime.now().isoformat()
                        }, websocket)
                    except Exception as e:
                        logger.error(f"Failed to fetch initial data: {e}")
                        await manager.send_personal_message({
                            'type': 'error',
                            'message': f'Failed to fetch data for symbols: {symbols}',
                            'timestamp': datetime.now().isoformat()
                        }, websocket)
                
                await manager.send_personal_message({
                    'type': 'subscribed',
                    'symbols': symbols,
                    'total_subscriptions': len(manager.subscriptions[websocket]),
                    'timestamp': datetime.now().isoformat()
                }, websocket)
                
            elif action == 'unsubscribe':
                symbols = [s.upper() for s in message.get('symbols', [])]
                
                # Remove from this connection's subscriptions
                manager.subscriptions[websocket].difference_update(symbols)
                
                # Unsubscribe from service if no other connections need these symbols
                for symbol in symbols:
                    still_needed = any(
                        symbol in subs for subs in manager.subscriptions.values()
                    )
                    if not still_needed:
                        realtime_service.unsubscribe_symbol(symbol)
                
                await manager.send_personal_message({
                    'type': 'unsubscribed',
                    'symbols': symbols,
                    'remaining_subscriptions': len(manager.subscriptions[websocket]),
                    'timestamp': datetime.now().isoformat()
                }, websocket)
                
            elif action == 'ping':
                await manager.send_personal_message({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
                
            elif action == 'get_subscriptions':
                await manager.send_personal_message({
                    'type': 'subscriptions',
                    'symbols': list(manager.subscriptions[websocket]),
                    'count': len(manager.subscriptions[websocket]),
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            else:
                await manager.send_personal_message({
                    'type': 'error',
                    'message': f'Unknown action: {action}',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/realtime/{symbols}")
async def get_realtime_quotes(symbols: str):
    """
    Get real-time quotes using the enhanced service (REST API)
    """
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbol_list) > 20:
        raise HTTPException(status_code=400, detail="Too many symbols (max 20 for real-time)")
    
    try:
        quotes = await realtime_service.get_real_time_quotes(symbol_list)
        return {
            "quotes": quotes,
            "count": len(quotes),
            "requested": len(symbol_list),
            "timestamp": datetime.now().isoformat(),
            "source": "realtime_api"
        }
    except Exception as e:
        logger.error(f"Failed to fetch real-time quotes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch real-time data")


@router.get("/status")
async def get_status():
    """
    Get real-time service status
    """
    return {
        "active_connections": len(manager.active_connections),
        "subscribed_symbols": list(realtime_service.get_subscribed_symbols()),
        "symbol_count": len(realtime_service.get_subscribed_symbols()),
        "service_status": "running",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/demo")
async def start_demo():
    """
    Start demo mode with popular stocks
    """
    demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    for symbol in demo_symbols:
        realtime_service.subscribe_symbol(symbol)
    
    return {
        "message": "Demo mode started",
        "symbols": demo_symbols,
        "instructions": {
            "websocket": "Connect to /ws and send: {\"action\": \"subscribe\", \"symbols\": [\"AAPL\"]}",
            "rest": "GET /realtime/AAPL,GOOGL,MSFT for instant quotes"
        },
        "timestamp": datetime.now().isoformat()
    }


# Initialize real-time updates when module loads
async def start_realtime_service():
    """Start the real-time update service"""
    await realtime_service.start_real_time_updates(
        symbols=[],  # Start with empty list, will be populated by subscriptions
        callback=broadcast_market_updates,
        interval_seconds=5  # Update every 5 seconds
    )
    logger.info("Real-time market service started")

# Note: In a real application, you'd call start_realtime_service() during app startup