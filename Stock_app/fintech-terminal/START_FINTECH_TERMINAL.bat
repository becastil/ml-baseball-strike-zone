@echo off
setlocal enabledelayedexpansion
title Fintech Terminal - Advanced Launcher

:: Set colors
color 0A

:: Display banner
cls
echo.
echo    ███████╗██╗███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗
echo    ██╔════╝██║████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║
echo    █████╗  ██║██╔██╗ ██║   ██║   █████╗  ██║     ███████║
echo    ██╔══╝  ██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║
echo    ██║     ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║
echo    ╚═╝     ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝
echo.
echo                 TERMINAL - Professional Trading Platform
echo    ════════════════════════════════════════════════════════
echo.

:: Check prerequisites
echo [*] Checking prerequisites...

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo    [X] Node.js is NOT installed
    echo.
    echo    Please install Node.js from: https://nodejs.org/
    echo    After installation, restart this launcher.
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo    [✓] Node.js !NODE_VERSION! found
)

:: Menu
:menu
echo.
echo    ════════════════════════════════════════════════════════
echo                        LAUNCH OPTIONS
echo    ════════════════════════════════════════════════════════
echo.
echo    [1] Start Frontend Only (Recommended)
echo    [2] Start Frontend + Backend (Requires Python)
echo    [3] Start with Docker (Requires Docker Desktop)
echo    [4] Install/Update Dependencies
echo    [5] Open in Browser
echo    [6] Exit
echo.
set /p choice="    Select option (1-6): "

if "%choice%"=="1" goto start_frontend
if "%choice%"=="2" goto start_full
if "%choice%"=="3" goto start_docker
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto open_browser
if "%choice%"=="6" exit /b 0

echo    Invalid selection. Please try again.
goto menu

:start_frontend
cls
echo.
echo    ═══════════════════════════════════════════════════════
echo              STARTING FRONTEND APPLICATION
echo    ═══════════════════════════════════════════════════════
echo.

cd /d "%~dp0frontend"

:: Check if dependencies are installed
if not exist "node_modules" (
    echo    [*] First time setup detected...
    echo    [*] Installing dependencies (this may take a few minutes)...
    echo.
    call npm install
    if %errorlevel% neq 0 (
        echo.
        echo    [X] Failed to install dependencies
        pause
        goto menu
    )
    echo.
    echo    [✓] Dependencies installed successfully!
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
echo    ═══════════════════════════════════════════════════════
echo    [✓] Starting application on port !PORT!
echo.
echo    The app will open in your browser automatically.
echo    If not, visit: http://localhost:!PORT!
echo.
echo    Press Ctrl+C to stop the application
echo    ═══════════════════════════════════════════════════════
echo.

:: Open browser after a delay
start /min cmd /c "timeout /t 5 >nul && start http://localhost:!PORT!"

:: Start the server
set VITE_PORT=!PORT!
call npm run dev -- --port !PORT!

pause
goto menu

:start_full
echo.
echo    [!] Full stack mode requires Python and PostgreSQL
echo    [!] This feature is coming soon...
echo.
pause
goto menu

:start_docker
echo.
echo    [*] Starting Docker containers...
cd /d "%~dp0"
docker-compose up -d
if %errorlevel% equ 0 (
    echo    [✓] Docker containers started successfully!
    echo    [*] Opening application...
    timeout /t 5 >nul
    start http://localhost:3000
) else (
    echo    [X] Failed to start Docker containers
    echo    Make sure Docker Desktop is running and WSL integration is enabled
)
pause
goto menu

:install_deps
echo.
echo    [*] Installing/Updating dependencies...
cd /d "%~dp0frontend"
call npm install
echo    [✓] Dependencies updated!
pause
goto menu

:open_browser
start http://localhost:3000
start http://localhost:3001
start http://localhost:3002
goto menu