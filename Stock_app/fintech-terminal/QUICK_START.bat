@echo off
:: Simple one-click launcher for Fintech Terminal

cd /d "%~dp0frontend"

:: Install dependencies if needed (silent)
if not exist "node_modules" (
    echo Installing dependencies, please wait...
    npm install --silent
)

:: Open browser after 5 seconds
start /min cmd /c "timeout /t 5 >nul && start http://localhost:3000"

:: Start the app
npm run dev