@echo off
echo 🚀 Starting Fintech Terminal Real-Time Demo
echo ==========================================

cd /d "%~dp0"

echo.
echo 📦 Installing dependencies...
cd frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
)

cd ..\backend
echo Installing backend dependencies...
pip install -r requirements.txt

echo.
echo 🔥 Starting services...
echo ==========================================

echo.
echo 🖥️  INSTRUCTIONS:
echo 1. Backend will start on http://localhost:8000
echo 2. Frontend will start on http://localhost:3000
echo 3. Open http://localhost:3000 in your browser
echo 4. Check browser console for WebSocket connection logs
echo 5. Watch the Real-Time Market Widget for live updates every 5 seconds!
echo.

echo Starting backend server...
start "Backend" cmd /k "cd /d %~dp0backend && uvicorn main:app --reload --host 0.0.0.0"

timeout /t 3

echo Starting frontend server...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ✅ Services started! 
echo 🌐 Open http://localhost:3000 to see real-time updates
echo 📡 WebSocket connection will be established automatically
echo.
pause