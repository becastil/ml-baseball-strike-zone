@echo off
echo 🔧 Quick Fix Installation for Fintech Terminal
echo =============================================

cd /d "%~dp0"

echo.
echo 🐍 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.9+ first
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 📦 Creating fresh virtual environment...
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo ⚡ Activating virtual environment...
call venv\Scripts\activate

echo.
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo 📥 Installing minimal backend dependencies...
cd backend

echo Trying minimal requirements first...
python -m pip install -r requirements-minimal.txt

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Minimal requirements failed. Trying individual packages...
    
    echo Installing FastAPI...
    pip install fastapi
    
    echo Installing Uvicorn...
    pip install uvicorn[standard]
    
    echo Installing Pydantic...
    pip install pydantic
    
    echo Installing yfinance...
    pip install yfinance
    
    echo Installing aiohttp...
    pip install aiohttp
    
    echo Installing python-dotenv...
    pip install python-dotenv
    
    echo Installing websockets...
    pip install websockets
)

echo.
echo 🧪 Testing installation...
python -c "import fastapi, uvicorn, yfinance, pydantic; print('✅ Core packages working!')"

if %errorlevel% neq 0 (
    echo ❌ Installation test failed
    echo.
    echo 🛠️  Troubleshooting suggestions:
    echo 1. Install Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo 2. Try running as Administrator
    echo 3. Check Python version (should be 3.9+)
    echo 4. See TROUBLESHOOT_INSTALLATION.md for detailed help
    pause
    exit /b 1
)

echo.
echo 📁 Installing frontend dependencies...
cd ..\frontend

if not exist "node_modules" (
    echo Installing npm packages...
    npm install
    
    if %errorlevel% neq 0 (
        echo ⚠️  npm install failed. Trying with --legacy-peer-deps...
        npm install --legacy-peer-deps
    )
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 🚀 Ready to start the application:
echo.
echo Option 1 - Start manually:
echo   Terminal 1: cd backend ^&^& python -m uvicorn main:app --reload
echo   Terminal 2: cd frontend ^&^& npm run dev
echo.
echo Option 2 - Use start script:
echo   Double-click START_REALTIME_DEMO.bat
echo.
echo 🌐 Then open: http://localhost:3000
echo.
pause