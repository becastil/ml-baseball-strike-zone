@echo off
setlocal enabledelayedexpansion
title Fintech Terminal - Real-Time Data

color 0A
cls
echo.
echo    ╔═══════════════════════════════════════════════════════╗
echo    ║        FINTECH TERMINAL - REAL-TIME DATA              ║
echo    ║          Starting Backend + Frontend                  ║
echo    ╚═══════════════════════════════════════════════════════╝
echo.

:: Check Python
echo [*] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [X] Python is not installed!
        echo     Please install Python from: https://python.org/
        pause
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)
echo [✓] Python found

:: Check Node
echo [*] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Node.js is not installed!
    echo     Please install from: https://nodejs.org/
    pause
    exit /b 1
)
echo [✓] Node.js found

:: Install Python dependencies if needed
echo.
echo [*] Setting up backend...
cd /d "%~dp0backend"

:: Check if pip is available
!PYTHON! -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Installing pip...
    !PYTHON! -m ensurepip --default-pip
)

:: Install simple requirements
echo [*] Installing backend dependencies...
!PYTHON! -m pip install -r requirements_simple.txt --quiet --disable-pip-version-check

:: Start backend in new window
echo [*] Starting backend API server...
start "Fintech Backend" cmd /k "!PYTHON! app/main_simple.py"

:: Wait for backend to start
echo [*] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

:: Setup frontend
echo.
echo [*] Setting up frontend...
cd /d "%~dp0frontend"

:: Install frontend dependencies if needed
if not exist "node_modules" (
    echo [*] Installing frontend dependencies...
    call npm install --silent
)

:: Find available port
set PORT=3000
netstat -an | findstr :!PORT! >nul
if %errorlevel% equ 0 (
    set PORT=3001
    netstat -an | findstr :!PORT! >nul
    if %errorlevel% equ 0 (
        set PORT=3002
    )
)

echo.
echo    ╔═══════════════════════════════════════════════════════╗
echo    ║              REAL-TIME DATA ACTIVE!                   ║
echo    ║                                                       ║
echo    ║   Backend API: http://localhost:8000                  ║
echo    ║   Frontend:    http://localhost:!PORT!                ║
echo    ║                                                       ║
echo    ║   The app will open automatically...                  ║
echo    ║                                                       ║
echo    ║   To stop: Close both terminal windows                ║
echo    ╚═══════════════════════════════════════════════════════╝
echo.

:: Open browser after delay
start /min cmd /c "timeout /t 8 >nul && start http://localhost:!PORT!"

:: Start frontend
npm run dev -- --port !PORT!