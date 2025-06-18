@echo off
:: Ultimate launcher for Fintech Terminal - Auto-detects best configuration

title Fintech Terminal Launcher
color 0A

echo.
echo ════════════════════════════════════════════════════════
echo    FINTECH TERMINAL - Smart Launcher
echo ════════════════════════════════════════════════════════
echo.

:: Check if Python is available for real data
python --version >nul 2>&1
if %errorlevel% equ 0 (
    goto :with_real_data
) else (
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        goto :with_real_data
    )
)

:: No Python - launch frontend only
echo [!] Python not found - Starting UI only (mock data)
echo.
cd /d "%~dp0frontend"
if not exist "node_modules" npm install --silent
start http://localhost:3000
npm run dev
exit /b

:with_real_data
echo [✓] Python found - Starting with REAL market data!
echo.
call "%~dp0START_WITH_REAL_DATA.bat"