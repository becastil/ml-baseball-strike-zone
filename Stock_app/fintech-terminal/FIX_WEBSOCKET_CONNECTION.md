# 🔧 Fix WebSocket Connection Issues

## Quick Diagnosis

When you see "Failed to create WebSocket connection", it usually means:
1. **Backend is not running** (most common)
2. **Backend is running on wrong port**
3. **WebSocket endpoint doesn't exist**
4. **CORS/firewall blocking connection**

## ✅ Step-by-Step Fix

### Step 1: Check if Backend is Running

**Open a new Command Prompt and run:**
```bash
# Test if backend is accessible
curl http://localhost:8000/health

# OR open in browser:
# http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Fintech Terminal API", 
  "version": "0.1.0"
}
```

**If you get an error:** Backend is not running → Go to Step 2
**If you get the response:** Backend is running → Go to Step 3

### Step 2: Start the Backend

**Option A - Use the Automated Script:**
```bash
# Double-click:
START_REALTIME_DEMO.bat
```

**Option B - Manual Start:**
```bash
# Open Command Prompt
cd fintech-terminal

# Activate virtual environment
venv\Scripts\activate

# Start backend
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Will watch for changes in these directories: ['/path/to/backend']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 3: Test WebSocket Endpoint

**Check if the WebSocket endpoint exists:**
```bash
# Open browser to:
http://localhost:8000/docs

# Look for:
# - /api/v1/realtime/ws (WebSocket endpoint)
# - /api/v1/realtime/status (Status endpoint)
```

### Step 4: Update Backend Routes

**The issue might be that the realtime routes aren't properly included.**

**Check backend/app/main.py includes the realtime router:**
```python
# This should be in main.py:
from app.api.v1 import auth, market_data, realtime_market

app.include_router(
    realtime_market.router,
    prefix="/api/v1/realtime", 
    tags=["real-time-market"]
)
```

### Step 5: Test WebSocket Connection Manually

**Open browser console (F12) and test:**
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8000/api/v1/realtime/ws');

ws.onopen = () => console.log('✅ WebSocket connected!');
ws.onerror = (error) => console.log('❌ WebSocket error:', error);
ws.onclose = (event) => console.log('🔌 WebSocket closed:', event.code, event.reason);

// Subscribe to market data
ws.onopen = () => {
    ws.send(JSON.stringify({
        action: 'subscribe',
        symbols: ['AAPL', 'GOOGL', 'MSFT']
    }));
};

ws.onmessage = (event) => {
    console.log('📡 Received:', JSON.parse(event.data));
};
```

## 🚨 Common Issues and Quick Fixes

### Issue 1: "Connection refused" or "Failed to connect"

**Cause:** Backend not running
**Fix:** 
```bash
# Start backend first:
cd fintech-terminal/backend
python -m uvicorn main:app --reload
```

### Issue 2: "404 Not Found" on WebSocket

**Cause:** WebSocket endpoint doesn't exist
**Fix:** Update main.py to include realtime router

### Issue 3: "CORS error"

**Cause:** Frontend and backend on different origins
**Fix:** Make sure backend allows frontend origin

### Issue 4: "Module not found" errors in backend

**Cause:** Missing realtime_market module
**Fix:** The realtime_market.py file might be missing

## 🔍 Debugging Steps

### Check Backend Logs
```bash
# Look for these in backend terminal:
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000

# Look for errors like:
ModuleNotFoundError: No module named 'app.api.v1.realtime_market'
ImportError: cannot import name 'realtime_market'
```

### Check Frontend Console (F12)
```bash
# Look for:
🔌 Connecting to WebSocket...
❌ WebSocket error: Error in connection establishment

# Or:
🎉 WebSocket connected successfully
📡 Subscribing to symbols: ['AAPL', 'GOOGL', 'MSFT']
```

### Check Network Tab (F12 → Network → WS)
- Should show WebSocket connection to `ws://localhost:8000/api/v1/realtime/ws`
- Connection should be green/open
- Messages should flow every 5 seconds

## 🛠️ Quick Fix Scripts

### Test Backend Connectivity
```bash
# Save as test_backend.py
import requests
import websockets
import asyncio

def test_http():
    try:
        response = requests.get('http://localhost:8000/health')
        print(f"✅ HTTP: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ HTTP Error: {e}")

async def test_websocket():
    try:
        async with websockets.connect('ws://localhost:8000/api/v1/realtime/ws') as ws:
            print("✅ WebSocket connected!")
            await ws.send('{"action": "ping"}')
            response = await ws.recv()
            print(f"📡 Response: {response}")
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")

if __name__ == "__main__":
    test_http()
    asyncio.run(test_websocket())
```

### Start Services in Correct Order
```bash
# Terminal 1 - Backend FIRST
cd fintech-terminal
venv\Scripts\activate
cd backend
python -m uvicorn main:app --reload

# Wait for "Application startup complete"

# Terminal 2 - Frontend SECOND  
cd fintech-terminal/frontend
npm run dev

# Wait for "ready in XXX ms"

# Terminal 3 - Test connection
python test_backend.py
```

## ✅ Success Indicators

You know it's working when:

1. **Backend terminal shows:**
   ```
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

2. **Browser at localhost:8000/docs shows:**
   - WebSocket endpoint: `/api/v1/realtime/ws`
   - Status endpoint: `/api/v1/realtime/status`

3. **Frontend console (F12) shows:**
   ```
   🔌 Connecting to WebSocket...
   🎉 WebSocket connected successfully
   📡 Subscribing to symbols: ['AAPL', 'GOOGL', 'MSFT']
   ```

4. **Dashboard shows:**
   - Green "Live" indicator
   - Stock prices updating every 5 seconds
   - Activity log with timestamps

## 🔄 If Still Not Working

Try this **complete restart sequence:**

1. **Stop everything** (Ctrl+C in all terminals)
2. **Start backend first:**
   ```bash
   cd fintech-terminal/backend
   python -m uvicorn main:app --reload
   ```
3. **Wait for "Application startup complete"**
4. **Test backend:** Open `http://localhost:8000/health`
5. **Start frontend:**
   ```bash
   cd fintech-terminal/frontend  
   npm run dev
   ```
6. **Test frontend:** Open `http://localhost:3000`

The key is **starting backend BEFORE frontend** and making sure each step completes successfully!