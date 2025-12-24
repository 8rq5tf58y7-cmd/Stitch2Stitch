@echo off
REM Stitch2Stitch Launcher for Windows
REM This script activates the virtual environment and launches the application

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run install_windows.ps1 or install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment and run
call venv\Scripts\activate.bat
python src\main.py %*

