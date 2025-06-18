# PowerShell launcher for Fintech Terminal
# Right-click and select "Run with PowerShell" or double-click if execution policy allows

$Host.UI.RawUI.WindowTitle = "Fintech Terminal"
$Host.UI.RawUI.BackgroundColor = "Black"
$Host.UI.RawUI.ForegroundColor = "Green"
Clear-Host

Write-Host @"

    ╔═══════════════════════════════════════════════════════╗
    ║           FINTECH TERMINAL - LAUNCHER                 ║
    ║        Professional Trading Platform                  ║
    ╚═══════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Check Node.js
Write-Host "[*] Checking Node.js..." -ForegroundColor Yellow
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "[X] Node.js is not installed!" -ForegroundColor Red
    Write-Host "    Please install from: https://nodejs.org/" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit
}

$nodeVersion = node --version
Write-Host "[✓] Node.js $nodeVersion found" -ForegroundColor Green

# Navigate to frontend
Set-Location "$PSScriptRoot\frontend"

# Install dependencies if needed
if (!(Test-Path "node_modules")) {
    Write-Host "`n[*] Installing dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "[✓] Dependencies installed!" -ForegroundColor Green
}

# Find available port
$port = 3000
while (Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -ErrorAction SilentlyContinue) {
    $port++
}

Write-Host "`n╔═══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Starting Fintech Terminal on port $port...          ║" -ForegroundColor Cyan
Write-Host "║                                                       ║" -ForegroundColor Cyan
Write-Host "║   App will open in your browser automatically        ║" -ForegroundColor Cyan
Write-Host "║   Or visit: http://localhost:$port                    ║" -ForegroundColor Cyan
Write-Host "║                                                       ║" -ForegroundColor Cyan
Write-Host "║   Press Ctrl+C to stop                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Open browser after delay
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:$using:port"
} | Out-Null

# Start the app
npm run dev -- --port $port