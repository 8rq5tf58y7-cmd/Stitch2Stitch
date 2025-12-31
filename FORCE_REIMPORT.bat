@echo off
echo ============================================================
echo   FORCE COMPLETE PYTHON REIMPORT
echo ============================================================
echo.
echo NOTE: This clears Python cache for code changes.
echo       To clear COLMAP feature cache (force re-extraction),
echo       delete the "colmap_workspace" folder in your image directory.
echo.

cd /d "%~dp0"

echo Clearing ALL Python cache...
for /d /r src %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q src\*.pyc 2>nul

echo Clearing cache from all subdirectories...
if exist "src\__pycache__" rmdir /s /q "src\__pycache__" 2>nul
if exist "src\gui\__pycache__" rmdir /s /q "src\gui\__pycache__" 2>nul
if exist "src\core\__pycache__" rmdir /s /q "src\core\__pycache__" 2>nul
if exist "src\utils\__pycache__" rmdir /s /q "src\utils\__pycache__" 2>nul
if exist "src\external\__pycache__" rmdir /s /q "src\external\__pycache__" 2>nul
if exist "src\ml\__pycache__" rmdir /s /q "src\ml\__pycache__" 2>nul

echo Using PowerShell to ensure complete cleanup...
powershell -Command "Get-ChildItem -Path '%~dp0src' -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force" 2>nul
powershell -Command "Get-ChildItem -Path '%~dp0src' -Filter *.pyc -Recurse -Force | Remove-Item -Force" 2>nul

echo.
echo Setting PYTHONDONTWRITEBYTECODE to prevent cache creation...
set PYTHONDONTWRITEBYTECODE=1

echo.
echo Activating venv...
set "VENV_PATH=%LOCALAPPDATA%\Stitch2Stitch\venv"
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo ERROR: No venv found
    pause
    exit /b 1
)

echo.
echo Launching with NO CACHE...
python -B src\main.py

echo.
echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application exited with error code: %ERRORLEVEL%
) else (
    echo Application exited normally
)

echo.
pause
