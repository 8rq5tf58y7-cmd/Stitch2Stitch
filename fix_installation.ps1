# Quick fix script to copy source files to installation directory
# Run this if you've already installed but the launcher can't find the source files

$InstallDir = "$env:LOCALAPPDATA\Stitch2Stitch"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Fixing installation..." -ForegroundColor Cyan
Write-Host ""

# Check if installation directory exists
if (-not (Test-Path $InstallDir)) {
    Write-Host "Installation directory not found: $InstallDir" -ForegroundColor Red
    Write-Host "Please run install_windows.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Copy source files
Write-Host "Copying source files..." -ForegroundColor Green
if (Test-Path (Join-Path $ScriptDir "src")) {
    $srcDest = Join-Path $InstallDir "src"
    if (Test-Path $srcDest) {
        Write-Host "Removing old src directory..." -ForegroundColor Yellow
        Remove-Item -Path $srcDest -Recurse -Force
    }
    Copy-Item -Path (Join-Path $ScriptDir "src") -Destination $InstallDir -Recurse -Force
    Write-Host "Source files copied successfully!" -ForegroundColor Green
} else {
    Write-Host "Error: Could not find src directory in: $ScriptDir" -ForegroundColor Red
    exit 1
}

# Copy other necessary files
Write-Host "Copying additional files..." -ForegroundColor Green
$filesToCopy = @("requirements.txt", "README.md")
foreach ($file in $filesToCopy) {
    $sourceFile = Join-Path $ScriptDir $file
    if (Test-Path $sourceFile) {
        Copy-Item -Path $sourceFile -Destination $InstallDir -Force
        Write-Host "  Copied: $file" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Installation fixed!" -ForegroundColor Green
Write-Host "You can now use launch_windows.bat or the desktop shortcut." -ForegroundColor Cyan
Write-Host ""

