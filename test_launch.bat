@echo off
REM Simple test script to diagnose launch issues
setlocal enabledelayedexpansion

echo ============================================================
echo LAUNCH DIAGNOSTICS
echo ============================================================
echo.
echo Current directory: %CD%
echo.

REM Check for venv
if exist "venv\Scripts\python.exe" (
    echo [OK] Found venv at: %CD%\venv
    set PYTHON_EXE=venv\Scripts\python.exe
) else if exist "%LOCALAPPDATA%\Stitch2Stitch\venv\Scripts\python.exe" (
    echo [OK] Found venv at: %LOCALAPPDATA%\Stitch2Stitch\venv
    set PYTHON_EXE=%LOCALAPPDATA%\Stitch2Stitch\venv\Scripts\python.exe
) else (
    echo [ERROR] No venv found!
    goto :end
)

echo.
echo Python executable: !PYTHON_EXE!
echo.

REM Check for main.py
if exist "src\main.py" (
    echo [OK] Found src\main.py
) else (
    echo [ERROR] src\main.py not found!
    goto :end
)

echo.
echo Testing Python version...
!PYTHON_EXE! --version
echo.

echo Testing imports...
!PYTHON_EXE! -c "import sys; print('Python path:', sys.executable)"
!PYTHON_EXE! -c "import PySide6; print('PySide6:', PySide6.__version__)"
!PYTHON_EXE! -c "import cv2; print('OpenCV:', cv2.__version__)"
echo.

echo All checks passed!
echo.

:end
echo ============================================================
echo.
echo Press any key to exit...
pause >nul
