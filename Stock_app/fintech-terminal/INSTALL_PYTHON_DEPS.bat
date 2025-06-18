@echo off
title Installing Python Dependencies for Fintech Terminal

echo ═══════════════════════════════════════════════════════
echo    Installing Python Dependencies
echo    This will install the required packages for real-time data
echo ═══════════════════════════════════════════════════════
echo.

cd /d "%~dp0backend"

:: Try python first, then python3
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python is not installed!
        echo.
        echo Please install Python from: https://python.org/
        echo Make sure to check "Add Python to PATH" during installation
        echo.
        pause
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)

echo Using %PYTHON%
echo.

:: Upgrade pip first
echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip

echo.
echo Installing required packages...
%PYTHON% -m pip install fastapi uvicorn yfinance pandas websockets python-multipart

echo.
echo ═══════════════════════════════════════════════════════
echo    Installation Complete!
echo    You can now run START_WITH_REAL_DATA.bat
echo ═══════════════════════════════════════════════════════
echo.
pause