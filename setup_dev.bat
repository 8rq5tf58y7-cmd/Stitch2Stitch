@echo off
REM Quick development setup - creates venv in project directory
REM Use this if you want to develop/run from the project directory

echo ========================================
echo Stitch2Stitch Development Setup
echo ========================================
echo.

cd /d "%~dp0"

REM Check if venv already exists
if exist "venv\Scripts\activate.bat" (
    echo Virtual environment already exists.
    echo.
    echo To activate:
    echo   venv\Scripts\activate
    echo.
    echo To run:
    echo   python src\main.py
    echo.
    pause
    exit /b 0
)

REM Check for Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found!
    echo Please install Python 3.9+ and add it to PATH.
    echo.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo Error creating virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing dependencies...
echo This may take several minutes...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Warning: requirements.txt not found
    echo Installing basic packages...
    pip install opencv-python opencv-contrib-python numpy scipy Pillow PyQt6 matplotlib tqdm pyyaml tifffile psutil
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run the application:
echo   1. Activate: venv\Scripts\activate
echo   2. Run: python src\main.py
echo.
echo Or use: launch_windows.bat
echo.
pause

