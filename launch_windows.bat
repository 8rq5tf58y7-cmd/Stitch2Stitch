@echo off
REM Stitch2Stitch Launcher for Windows
REM This script activates the virtual environment and launches the application

cd /d "%~dp0"

REM Check for virtual environment in multiple locations
set VENV_FOUND=0
set VENV_PATH=

REM First, check if venv exists in current directory (project directory)
if exist "venv\Scripts\activate.bat" (
    set VENV_PATH=venv
    set VENV_FOUND=1
    goto :found_venv
)

REM Second, check if installed in AppData\Local\Stitch2Stitch
set INSTALL_DIR=%LOCALAPPDATA%\Stitch2Stitch
if exist "%INSTALL_DIR%\venv\Scripts\activate.bat" (
    set VENV_PATH=%INSTALL_DIR%\venv
    set VENV_FOUND=1
    goto :found_venv
)

REM If not found, show error
if %VENV_FOUND%==0 (
    echo Virtual environment not found!
    echo.
    echo Checked locations:
    echo   1. %CD%\venv
    echo   2. %INSTALL_DIR%\venv
    echo.
    echo Please run install_windows.ps1 or install.bat first.
    echo.
    pause
    exit /b 1
)

:found_venv
REM Activate virtual environment and run
call "%VENV_PATH%\Scripts\activate.bat"

REM Clear Python cache to avoid import issues
if exist "src\__pycache__" (
    rmdir /s /q "src\__pycache__" 2>nul
)
if exist "src\gui\__pycache__" (
    rmdir /s /q "src\gui\__pycache__" 2>nul
)
if exist "src\core\__pycache__" (
    rmdir /s /q "src\core\__pycache__" 2>nul
)
if exist "src\utils\__pycache__" (
    rmdir /s /q "src\utils\__pycache__" 2>nul
)

REM Check if we're in the project directory or need to change
if exist "src\main.py" (
    REM We're in the project directory - run directly (main.py sets up path)
    python src\main.py %*
) else if exist "%INSTALL_DIR%\src\main.py" (
    REM We're running from installed location
    cd /d "%INSTALL_DIR%"
    REM Clear cache there too
    if exist "src\__pycache__" (
        rmdir /s /q "src\__pycache__" 2>nul
    )
    python src\main.py %*
) else (
    echo Error: Could not find src\main.py
    echo Current directory: %CD%
    pause
    exit /b 1
)

