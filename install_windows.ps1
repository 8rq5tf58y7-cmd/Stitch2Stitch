# Stitch2Stitch Windows Installer
# PowerShell script to install Python and all dependencies

param(
    [string]$PythonVersion = "3.11.0",
    [string]$InstallDir = "$env:LOCALAPPDATA\Stitch2Stitch"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stitch2Stitch Windows Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Warning: Not running as Administrator. Some features may require elevation." -ForegroundColor Yellow
}

# Create installation directory
Write-Host "Creating installation directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Set-Location $InstallDir

# Check if Python is installed
Write-Host "Checking for Python installation..." -ForegroundColor Green
$pythonInstalled = $false
$pythonPath = $null

# Check common Python locations
$pythonPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "$env:ProgramFiles\Python310\python.exe",
    "$env:ProgramFiles\Python39\python.exe",
    "C:\Python311\python.exe",
    "C:\Python310\python.exe",
    "C:\Python39\python.exe"
)

foreach ($path in $pythonPaths) {
    if (Test-Path $path) {
        $pythonPath = $path
        $pythonInstalled = $true
        Write-Host "Found Python at: $pythonPath" -ForegroundColor Green
        break
    }
}

# Also check PATH
if (-not $pythonInstalled) {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonPath = "python"
            $pythonInstalled = $true
            Write-Host "Found Python in PATH: $pythonVersion" -ForegroundColor Green
        }
    } catch {
        # Python not in PATH
    }
}

# Download and install Python if not found
if (-not $pythonInstalled) {
    Write-Host "Python not found. Downloading Python installer..." -ForegroundColor Yellow
    
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    $pythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    
    try {
        Write-Host "Downloading from: $pythonUrl" -ForegroundColor Cyan
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing
        
        Write-Host "Installing Python..." -ForegroundColor Green
        Write-Host "Please follow the Python installer prompts." -ForegroundColor Yellow
        Write-Host "IMPORTANT: Check 'Add Python to PATH' during installation!" -ForegroundColor Red
        
        Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        # Verify installation
        Start-Sleep -Seconds 2
        $pythonPath = "python"
        
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            Write-Host "Python installation may need a restart. Please restart your terminal and run this script again." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Python installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Error downloading/installing Python: $_" -ForegroundColor Red
        Write-Host "Please install Python manually from https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
}

# Verify Python version
Write-Host "Verifying Python version..." -ForegroundColor Green
$version = & $pythonPath --version
Write-Host "Python version: $version" -ForegroundColor Cyan

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
& $pythonPath -m venv venv

if (-not (Test-Path "venv\Scripts\activate.ps1")) {
    Write-Host "Error creating virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& "venv\Scripts\python.exe" -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow

$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $requirementsPath) {
    & "venv\Scripts\pip.exe" install -r $requirementsPath
} else {
    Write-Host "requirements.txt not found. Installing default packages..." -ForegroundColor Yellow
    & "venv\Scripts\pip.exe" install opencv-python opencv-contrib-python numpy scipy Pillow PyQt6 matplotlib tqdm pyyaml tifffile psutil
}

# Create launcher script
Write-Host "Creating launcher..." -ForegroundColor Green
$launcherScript = @"
@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
python src\main.py %*
"@

$launcherScript | Out-File -FilePath "Stitch2Stitch.bat" -Encoding ASCII

# Create desktop shortcut (optional)
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Stitch2Stitch.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $InstallDir "Stitch2Stitch.bat"
$shortcut.WorkingDirectory = $InstallDir
$shortcut.Description = "Stitch2Stitch - Advanced Panoramic Image Stitching"
$shortcut.Save()

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation directory: $InstallDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run Stitch2Stitch:" -ForegroundColor Yellow
Write-Host "  1. Double-click Stitch2Stitch.bat" -ForegroundColor White
Write-Host "  2. Or run: cd `"$InstallDir`" && Stitch2Stitch.bat" -ForegroundColor White
Write-Host ""
Write-Host "Desktop shortcut created!" -ForegroundColor Green
Write-Host ""

