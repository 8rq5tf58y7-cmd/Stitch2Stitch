@echo off
REM Stitch2Stitch Launcher for Windows
REM This script activates the virtual environment and launches the application

echo Starting Stitch2Stitch...
cd /d "%~dp0"
echo Current directory: %CD%

REM Enable delayed expansion for proper variable handling in conditionals
setlocal enabledelayedexpansion

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
set "INSTALL_DIR=%LOCALAPPDATA%\Stitch2Stitch"
echo Checking: !INSTALL_DIR!\venv\Scripts\activate.bat
if exist "!INSTALL_DIR!\venv\Scripts\activate.bat" (
    echo Found venv in AppData
    set "VENV_PATH=!INSTALL_DIR!\venv"
    set VENV_FOUND=1
    goto :found_venv
)

REM If not found, show error
if !VENV_FOUND!==0 (
    echo Virtual environment not found!
    echo.
    echo Checked locations:
    echo   1. %CD%\venv
    echo   2. !INSTALL_DIR!\venv
    echo.
    echo Please run install_windows.ps1 or install.bat first.
    echo.
    pause
    exit /b 1
)

:found_venv
REM Activate virtual environment and run
echo Activating virtual environment: !VENV_PATH!
call "!VENV_PATH!\Scripts\activate.bat"

REM Add COLMAP from hlocenv conda environment to PATH
set "HLOCENV_PATH=%USERPROFILE%\miniconda3\envs\hlocenv"
if exist "!HLOCENV_PATH!\Library\bin\colmap.exe" (
    set "PATH=!HLOCENV_PATH!\Library\bin;!PATH!"
    echo COLMAP found in hlocenv environment
)

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
if exist "src\external\__pycache__" (
    rmdir /s /q "src\external\__pycache__" 2>nul
)
if exist "src\ml\__pycache__" (
    rmdir /s /q "src\ml\__pycache__" 2>nul
)

REM Check if we're in the project directory or need to change
echo Checking for src\main.py...
if exist "src\main.py" (
    echo [DEBUG] Found src\main.py, running from project directory...
    REM We're in the project directory - run directly (main.py sets up path)
    echo [DEBUG] Executing: python src\main.py
    python src\main.py %*
    set EXIT_CODE=!ERRORLEVEL!
    echo [DEBUG] Python exited with code: !EXIT_CODE!
    echo [DEBUG] About to jump to show_logs...
    goto :show_logs
) else if exist "!INSTALL_DIR!\src\main.py" (
    echo [DEBUG] Running from installed location: !INSTALL_DIR!
    REM We're running from installed location
    cd /d "!INSTALL_DIR!"
    REM Clear cache there too
    if exist "src\__pycache__" (
        rmdir /s /q "src\__pycache__" 2>nul
    )
    python src\main.py %*
    set EXIT_CODE=!ERRORLEVEL!
    echo Python exited with code: !EXIT_CODE!
    goto :show_logs
) else (
    echo Error: Could not find src\main.py
    echo Current directory: %CD%
    pause
    exit /b 1
)

:show_logs
echo [DEBUG] Reached show_logs section
REM Show exit status
echo.
echo ============================================================
if !EXIT_CODE! EQU 0 (
    echo Application exited normally.
) else (
    echo Application exited with error code: !EXIT_CODE!
)
echo ============================================================

REM Show debug logs if available
echo [DEBUG] Checking for debug log...
if exist ".cursor\debug.log" (
    echo.
    echo Debug log available: .cursor\debug.log
    echo.
    echo Last 20 log entries:
    echo ------------------------------------------------------------
    powershell -Command "Get-Content '.cursor\debug.log' -Tail 20"
    echo ------------------------------------------------------------
) else (
    echo [DEBUG] No debug log found
)

echo.
echo [DEBUG] About to pause...
echo Press any key to close this window...
pause >nul
echo [DEBUG] Pause completed, script will now exit

REM Script will exit here after user presses a key

