#!/usr/bin/env python3
"""
Ultra-Simple Test Server - NO DEPENDENCIES REQUIRED
This server uses only Python standard library
"""
import http.server
import socketserver
import json
import urllib.parse
from datetime import datetime
import random

class FinTechHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to add timestamps and emojis"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 📡 {format % args}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        # Parse URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        # Set CORS headers
        def send_json_response(data, status=200):
            self.send_response(status)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            response = json.dumps(data, indent=2)
            self.wfile.write(response.encode())
            print(f"📤 Sent: {len(response)} bytes")
        
        # Route handlers
        if path == '/health':
            send_json_response({
                "status": "healthy",
                "service": "FinTech Terminal Test API",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "python_server": "Built-in HTTP Server"
            })
        
        elif path == '/':
            send_json_response({
                "message": "🚀 FinTech Terminal Test API",
                "status": "running",
                "endpoints": {
                    "health": "/health",
                    "market_quotes": "/api/v1/market/quotes",
                    "single_quote": "/api/v1/market/quote/AAPL",
                    "search": "/api/v1/market/search?query=apple"
                },
                "timestamp": datetime.now().isoformat()
            })
        
        elif path == '/api/v1/market/quotes':
            # Generate mock market data with realistic price movements
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
            base_prices = {"AAPL": 150, "GOOGL": 2750, "MSFT": 310, "TSLA": 240, "AMZN": 3200, "NVDA": 450, "META": 280}
            
            quotes = []
            for symbol in symbols:
                base_price = base_prices.get(symbol, 100)
                # Add some realistic price variation
                price_change = random.uniform(-2, 2)
                current_price = base_price + price_change
                change_percent = (price_change / base_price) * 100
                
                quotes.append({
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "change": round(price_change, 2),
                    "changePercent": round(change_percent, 2),
                    "volume": random.randint(1000000, 50000000),
                    "timestamp": datetime.now().isoformat(),
                    "source": "test_server"
                })
            
            send_json_response({
                "quotes": quotes,
                "count": len(quotes),
                "timestamp": datetime.now().isoformat(),
                "note": "📊 This is mock data for testing"
            })
        
        elif path.startswith('/api/v1/market/quote/'):
            # Single quote endpoint
            symbol = path.split('/')[-1].upper()
            base_prices = {"AAPL": 150, "GOOGL": 2750, "MSFT": 310, "TSLA": 240}
            base_price = base_prices.get(symbol, 100)
            
            price_change = random.uniform(-2, 2)
            current_price = base_price + price_change
            change_percent = (price_change / base_price) * 100
            
            send_json_response({
                "symbol": symbol,
                "name": f"{symbol} Inc.",
                "price": round(current_price, 2),
                "change": round(price_change, 2),
                "changePercent": round(change_percent, 2),
                "volume": random.randint(1000000, 50000000),
                "marketCap": random.randint(100000000000, 2000000000000),
                "timestamp": datetime.now().isoformat(),
                "source": "test_server"
            })
        
        elif path == '/api/v1/market/search':
            query_param = query.get('query', [''])[0].lower()
            
            # Mock search results
            all_stocks = [
                {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
                {"symbol": "TSLA", "name": "Tesla, Inc.", "exchange": "NASDAQ"},
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
            ]
            
            # Filter based on query
            results = [
                stock for stock in all_stocks
                if query_param in stock["symbol"].lower() or query_param in stock["name"].lower()
            ]
            
            send_json_response({
                "results": results,
                "count": len(results),
                "query": query_param,
                "timestamp": datetime.now().isoformat()
            })
        
        elif path == '/api/v1/realtime/status':
            send_json_response({
                "service_status": "running",
                "active_connections": 0,
                "message": "WebSocket not available in test server",
                "timestamp": datetime.now().isoformat()
            })
        
        else:
            send_json_response({
                "error": "Not Found",
                "path": path,
                "available_endpoints": ["/health", "/api/v1/market/quotes", "/api/v1/market/quote/{symbol}"]
            }, 404)

def main():
    PORT = 8000
    
    print("🚀 FinTech Terminal Test Server")
    print("=" * 50)
    print(f"🌐 Server URL: http://localhost:{PORT}")
    print("📊 Available endpoints:")
    print(f"  • Health Check: http://localhost:{PORT}/health")
    print(f"  • Market Quotes: http://localhost:{PORT}/api/v1/market/quotes")
    print(f"  • Single Quote: http://localhost:{PORT}/api/v1/market/quote/AAPL")
    print(f"  • Search: http://localhost:{PORT}/api/v1/market/search?query=apple")
    print(f"  • API Info: http://localhost:{PORT}/")
    print()
    print("🔧 Features:")
    print("  ✅ CORS enabled for frontend")
    print("  ✅ Mock market data with price changes")
    print("  ✅ JSON API responses")
    print("  ✅ No external dependencies")
    print()
    print("🛑 Press Ctrl+C to stop server")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), FinTechHandler) as httpd:
            print(f"✅ Server started successfully on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use!")
            print("💡 Solutions:")
            print("  1. Stop any other servers running on port 8000")
            print("  2. Or change PORT = 8001 in this script")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()