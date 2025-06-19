@echo off
echo 🔧 FinTech Terminal Backend Fix Script
echo =====================================
echo.

cd /d "%~dp0"

echo 📍 Current directory: %CD%
echo.

echo 🧪 TESTING ULTRA-SIMPLE SERVER (No Dependencies)
echo.
echo This server requires ZERO external packages!
echo It will work with any Python installation.
echo.

cd backend

echo 🚀 Starting test server on http://localhost:8000
echo.
echo 📊 Available endpoints once started:
echo   • Health: http://localhost:8000/health
echo   • Quotes: http://localhost:8000/api/v1/market/quotes
echo   • Search: http://localhost:8000/api/v1/market/search?query=apple
echo.
echo 🛑 Press Ctrl+C to stop the server
echo ✅ If this works, your Python setup is fine!
echo.

pause

echo Starting server...
python test_server.py

echo.
echo 📋 RESULTS:
echo.
if %errorlevel% equ 0 (
    echo ✅ Test server worked! Python setup is good.
    echo.
    echo 🚀 Next steps:
    echo 1. Try the FastAPI version: python working_fastapi.py
    echo 2. Install FastAPI: pip install fastapi uvicorn
    echo 3. If that works, connect your frontend
) else (
    echo ❌ Test server failed!
    echo.
    echo 🔍 Possible issues:
    echo 1. Python not installed or not in PATH
    echo 2. Port 8000 already in use
    echo 3. Firewall blocking the connection
    echo.
    echo 💡 Solutions:
    echo 1. Check: python --version
    echo 2. Check: netstat -an ^| findstr :8000
    echo 3. Run as Administrator
)

echo.
pause