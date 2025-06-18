@echo off
title Fintech Terminal - Starting Application

echo ===============================================
echo         FINTECH TERMINAL LAUNCHER
echo ===============================================
echo.

:: Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [1/3] Starting Frontend Application...
echo.

:: Navigate to frontend directory and start the dev server
cd /d "%~dp0frontend"

:: Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    echo.
)

:: Start the frontend server
echo Starting frontend server...
echo.
echo ===============================================
echo    Application will be available at:
echo    http://localhost:3000
echo.
echo    If port 3000 is busy, try:
echo    http://localhost:3001
echo    http://localhost:3002
echo ===============================================
echo.
echo Press Ctrl+C to stop the application
echo.

:: Run the development server
npm run dev

:: Keep window open if server stops
pause