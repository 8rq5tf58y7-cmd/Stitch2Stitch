# Stitch2Stitch Launcher for Windows (PowerShell)
# Right-click this file and select "Run with PowerShell"

Write-Host "Starting Stitch2Stitch..." -ForegroundColor Cyan

# Unique run id for debug logs (avoids needing to clear debug.log)
$env:STITCH2STITCH_RUN_ID = "run-" + (Get-Date).ToString("yyyyMMdd-HHmmss")

# Change to script directory
Set-Location $PSScriptRoot

# Check for venv in AppData
$installDir = "$env:LOCALAPPDATA\Stitch2Stitch"
$venvPath = "$installDir\venv"

if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Host "Found venv in AppData" -ForegroundColor Green
    $pythonExe = "$venvPath\Scripts\python.exe"
} elseif (Test-Path "venv\Scripts\python.exe") {
    Write-Host "Found venv in project directory" -ForegroundColor Green
    $pythonExe = "venv\Scripts\python.exe"
} else {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Checked locations:"
    Write-Host "  1. $venvPath"
    Write-Host "  2. $PSScriptRoot\venv"
    Write-Host ""
    Write-Host "Please run install_windows.ps1 first."
    Read-Host "Press Enter to exit"
    exit 1
}

# Add COLMAP from hlocenv to PATH
$hlocenvPath = "$env:USERPROFILE\miniconda3\envs\hlocenv\Library\bin"
if (Test-Path "$hlocenvPath\colmap.exe") {
    $env:PATH = "$hlocenvPath;$env:PATH"
    Write-Host "COLMAP found in hlocenv" -ForegroundColor Green
}

# Find main.py
if (Test-Path "src\main.py") {
    $mainPy = "src\main.py"
} elseif (Test-Path "$installDir\src\main.py") {
    Set-Location $installDir
    $mainPy = "src\main.py"
} else {
    Write-Host "ERROR: Could not find src\main.py" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Launching application..." -ForegroundColor Cyan
Write-Host ""

# Run the application and capture exit code
try {
    & $pythonExe $mainPy
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "ERROR: Failed to launch Python: $_" -ForegroundColor Red
    $exitCode = 1
}

# Always show exit status
Write-Host ""
Write-Host "============================================================" -ForegroundColor White
if ($exitCode -eq 0) {
    Write-Host "Application exited normally." -ForegroundColor Green
} else {
    Write-Host "Application exited with error code: $exitCode" -ForegroundColor Red
}
Write-Host "============================================================" -ForegroundColor White

# Check for debug logs
$debugLog = ".cursor\debug.log"
if (Test-Path $debugLog) {
    $logSize = (Get-Item $debugLog).Length
    Write-Host "`nDebug log available: $debugLog ($logSize bytes)" -ForegroundColor Cyan

    # Show last 20 lines of debug log
    Write-Host "`nLast 20 log entries:" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Get-Content $debugLog -Tail 20 | ForEach-Object { Write-Host "  $_" }
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
} else {
    Write-Host "`nNo debug log found at: $debugLog" -ForegroundColor Yellow
}

Write-Host "`nPress Enter to close this window..." -ForegroundColor Cyan
$null = Read-Host

